import json
from pathlib import Path

import numpy as np
import torch

from skill_pool.geodesic import SolverConfig, decoder_jacobians
from skill_pool.solver_audit import (
    LinearToy,
    ReluToy,
    SmoothToy,
    energy_gradient,
    finite_difference_gradient,
    linear_path,
    native_lbfgs_control,
    relu_path_diagnostics,
    run_toy_case,
)


ROOT = Path(__file__).resolve().parents[1]


def test_linear_toy_has_constant_metric_and_exact_log():
    model = LinearToy()
    start = np.full(8, 0.2)
    end = np.linspace(0.35, 1.05, 8)
    path = linear_path(start, end, 8)
    jacobians = decoder_jacobians(model, path)
    torch.testing.assert_close(jacobians, jacobians[:1].expand_as(jacobians))
    _, gradient = energy_gradient(model, path)
    assert np.abs(gradient).max() < 1e-10
    result, solved = run_toy_case(
        model, start, end, SolverConfig(segments=8, max_iterations=20), 1
    )
    assert result["converged"] and result["decoder_weights_unchanged"]
    assert result["log_error_l2"] < 1e-9
    np.testing.assert_allclose(solved, path.numpy(), atol=1e-12)


def test_smooth_native_lbfgs_converges_and_discretizations_are_close():
    model = SmoothToy()
    start = np.full(8, -0.7)
    end = np.linspace(0.4, 1.3, 8)
    config = dict(max_iterations=250, gradient_tolerance=1e-5)
    l8 = native_lbfgs_control(model, start, end, SolverConfig(segments=8, **config))
    l16 = native_lbfgs_control(model, start, end, SolverConfig(segments=16, **config))
    assert l8["final_energy"] < l8["initial_energy"]
    assert l16["final_energy"] < l16["initial_energy"]
    assert l8["converged_under_current_threshold"]
    assert l16["converged_under_current_threshold"]
    log8, log16 = np.asarray(l8["log_vector"]), np.asarray(l16["log_vector"])
    assert np.linalg.norm(log8 - log16) / np.linalg.norm(log16) < 0.15
    assert abs(l8["final_energy"] - l16["final_energy"]) / l16["final_energy"] < 0.01


def test_relu_same_region_and_boundary_diagnostics():
    model = ReluToy()
    same = relu_path_diagnostics(
        model, linear_path(np.full(8, 2.0), np.full(8, 2.2), 8).numpy()
    )
    crossing = relu_path_diagnostics(
        model, linear_path(np.full(8, -1.0), np.full(8, 1.0), 8).numpy()
    )
    assert same["boundary_segment_count"] == 0
    assert crossing["boundary_segment_count"] > 0
    assert len(crossing["activation_patterns"]) == 9
    assert np.asarray(crossing["jacobians"]).shape == (9, 60, 8)
    assert len(crossing["raw_gradient_infinity_norm_per_internal_node"]) == 7


def test_all_toy_energy_gradients_match_central_finite_difference():
    cases = [
        (LinearToy(), np.full(8, 0.2), np.linspace(0.35, 1.05, 8)),
        (SmoothToy(), np.full(8, -0.7), np.linspace(0.4, 1.3, 8)),
        (ReluToy(), np.full(8, 2.0), np.full(8, 2.2)),
        (ReluToy(), np.full(8, -1.0), np.full(8, 1.0)),
    ]
    for model, start, end in cases:
        result = finite_difference_gradient(model, linear_path(start, end, 4))
        assert result["passed"]
        assert result["max_absolute_error"] < 1e-7


def test_formal_audit_artifacts_and_fixed_real_sample_fields():
    audit = json.loads((ROOT / "solver_audit/audit_report.json").read_text())
    gradients = json.loads((ROOT / "solver_audit/gradient_check.json").read_text())
    assert gradients["all_passed"]
    assert audit["protected_artifacts_unchanged"]
    assert [row["choice"] for row in audit["classification"] if row["selected"]] == ["A", "B", "C"]
    rows = audit["real_samples"]
    assert len(rows) == 10
    assert [row["source_index"] for row in rows] == np.load(
        ROOT / "log_experiment/candidate_indices.npy"
    )[:10].tolist()
    required = {
        "initial_energy", "final_energy", "initial_raw_gradient_infinity_norm",
        "initial_normalized_gradient_infinity_norm", "final_raw_gradient_infinity_norm",
        "final_normalized_gradient_infinity_norm", "max_path_node_movement",
        "energy_relative_decrease", "lbfgs_step_count", "closure_evaluation_count",
    }
    assert all(required <= row.keys() for row in rows)
    for name in (
        "toy_energy_curves.png", "relu_activation_boundaries.png",
        "real_sample_residuals.png",
    ):
        assert (ROOT / "solver_audit" / name).read_bytes().startswith(b"\x89PNG")
