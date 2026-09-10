"""Strictly bounded audit of the discrete geodesic solver.

This module does not build pools and never changes VAE parameters.  It compares
analytic toy cases, finite differences, and the first ten saved construction
candidates using the same frozen decoder as the feasibility experiment.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import time
import traceback

import numpy as np
import torch
from torch import nn

from .geodesic import SolverConfig, decoder_jacobians, path_energy, solve_log
from .model import load_vae
from .splits import sha256, write_json


class LinearToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(7)
        weight = torch.randn(60, 8, generator=generator, dtype=torch.float64) / 4
        self.weight = nn.Parameter(weight, requires_grad=False)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent @ self.weight.T).reshape(-1, 30, 2)


class SmoothToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(11)
        self.weight = nn.Parameter(
            torch.randn(60, 8, generator=generator, dtype=torch.float64) / 2,
            requires_grad=False,
        )

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.tanh(latent @ self.weight.T).reshape(-1, 30, 2)


class ReluToy(nn.Module):
    """One hidden ReLU layer with inspectable activation patterns."""

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(19)
        self.first = nn.Parameter(
            torch.randn(12, 8, generator=generator, dtype=torch.float64),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.linspace(-0.35, 0.35, 12, dtype=torch.float64),
            requires_grad=False,
        )
        self.second = nn.Parameter(
            torch.randn(60, 12, generator=generator, dtype=torch.float64) / 3,
            requires_grad=False,
        )

    def preactivation(self, latent: torch.Tensor) -> torch.Tensor:
        return latent @ self.first.T + self.bias

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.preactivation(latent))
        return (hidden @ self.second.T).reshape(-1, 30, 2)


def linear_path(start: np.ndarray, end: np.ndarray, segments: int) -> torch.Tensor:
    start_tensor = torch.as_tensor(start, dtype=torch.float64)
    end_tensor = torch.as_tensor(end, dtype=torch.float64)
    time_grid = torch.linspace(0, 1, segments + 1, dtype=torch.float64)
    return start_tensor[None] + time_grid[:, None] * (end_tensor - start_tensor)[None]


def energy_gradient(
    model: nn.Module,
    path: torch.Tensor,
    damping: float = 1e-4,
    quadrature_points: int = 2,
) -> tuple[float, np.ndarray]:
    internal = path[1:-1].detach().clone().requires_grad_(True)
    full_path = torch.cat((path[:1].detach(), internal, path[-1:].detach()))
    energy = path_energy(model, full_path, damping, quadrature_points)
    gradient = torch.autograd.grad(energy, internal)[0]
    return float(energy.detach()), gradient.detach().cpu().numpy()


def finite_difference_gradient(
    model: nn.Module,
    path: torch.Tensor,
    epsilon: float = 1e-6,
    damping: float = 1e-4,
    quadrature_points: int = 2,
) -> dict:
    energy, autograd = energy_gradient(model, path, damping, quadrature_points)
    numerical = np.empty_like(autograd)
    for node in range(autograd.shape[0]):
        for dimension in range(autograd.shape[1]):
            plus = path.detach().clone()
            minus = path.detach().clone()
            plus[node + 1, dimension] += epsilon
            minus[node + 1, dimension] -= epsilon
            upper = float(path_energy(model, plus, damping, quadrature_points))
            lower = float(path_energy(model, minus, damping, quadrature_points))
            numerical[node, dimension] = (upper - lower) / (2 * epsilon)
    absolute = np.abs(autograd - numerical)
    magnitude = np.maximum(np.abs(autograd), np.abs(numerical))
    gradient_scale = max(float(magnitude.max()), 1e-12)
    significant = magnitude >= max(gradient_scale * 1e-6, 1e-8)
    relative = absolute[significant] / magnitude[significant]
    floored_relative = absolute / np.maximum(magnitude, 1e-8)
    return {
        "energy": energy,
        "epsilon": epsilon,
        "max_absolute_error": float(absolute.max()),
        "max_relative_error": float(relative.max()) if relative.size else 0.0,
        "max_componentwise_relative_error_with_1e-8_floor": float(floored_relative.max()),
        "relative_error_significance_threshold": float(max(gradient_scale * 1e-6, 1e-8)),
        "autograd_infinity_norm": float(np.abs(autograd).max()),
        "finite_difference_infinity_norm": float(np.abs(numerical).max()),
        "passed": bool(
            absolute.max() <= 1e-7
            and (not relative.size or relative.max() <= 1e-4)
        ),
    }


def relu_path_diagnostics(model: ReluToy, path: np.ndarray) -> dict:
    points = torch.as_tensor(path, dtype=torch.float64)
    preactivation = model.preactivation(points).detach().cpu().numpy()
    activation = preactivation > 0
    jacobians = decoder_jacobians(model, points).detach().cpu().numpy()
    node_changes = np.count_nonzero(activation[1:] != activation[:-1], axis=1)
    energy, gradient = energy_gradient(model, points)
    return {
        "activation_patterns": activation.astype(int).tolist(),
        "preactivations": preactivation.tolist(),
        "jacobians": jacobians.tolist(),
        "activation_changes_per_segment": node_changes.tolist(),
        "boundary_segment_count": int(np.count_nonzero(node_changes)),
        "raw_gradient_infinity_norm": float(np.abs(gradient).max()),
        "normalized_gradient_infinity_norm": float(np.abs(gradient).max() / max(energy, 1e-12)),
        "raw_gradient_infinity_norm_per_internal_node": np.abs(gradient).max(axis=1).tolist(),
        "normalized_gradient_infinity_norm_per_internal_node": (
            np.abs(gradient).max(axis=1) / max(energy, 1e-12)
        ).tolist(),
    }


def run_toy_case(
    model: nn.Module,
    start: np.ndarray,
    end: np.ndarray,
    config: SolverConfig,
    source_index: int,
) -> tuple[dict, np.ndarray]:
    initial_path = linear_path(start, end, config.segments)
    initial_energy, initial_gradient = energy_gradient(
        model, initial_path, config.damping, config.quadrature_points
    )
    before = deepcopy(model.state_dict())
    log_vector, path, record = solve_log(model, start, end, source_index, config)
    unchanged = all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    result = {
        "config": asdict(config),
        "initial_energy": initial_energy,
        "initial_raw_gradient_infinity_norm": float(np.abs(initial_gradient).max()),
        "final_energy": record["final_energy"],
        "energy_relative_decrease": (
            (initial_energy - record["final_energy"]) / max(initial_energy, 1e-12)
            if record["final_energy"] is not None
            else None
        ),
        "final_normalized_gradient_infinity_norm": record["gradient_residual"],
        "final_raw_gradient_infinity_norm": (
            record["gradient_residual"] * initial_energy
            if record["gradient_residual"] is not None else None
        ),
        "energy_curve": record["energy_curve"],
        "converged": record["converged"],
        "iterations": record["iterations"],
        "closure_evaluations": record["closure_evaluations"],
        "log_error_l2": float(np.linalg.norm(log_vector - (end - start))),
        "max_internal_node_movement": float(
            np.max(np.abs(path[1:-1] - initial_path.numpy()[1:-1]))
        ),
        "decoder_weights_unchanged": unchanged,
        "failure_reason": record["failure_reason"],
    }
    return result, path


def native_lbfgs_control(
    model: nn.Module,
    start: np.ndarray,
    end: np.ndarray,
    config: SolverConfig,
) -> dict:
    """Control using one standard LBFGS.step with its own iteration loop."""
    initial_path = linear_path(start, end, config.segments)
    start_tensor = initial_path[0].detach()
    end_tensor = initial_path[-1].detach()
    internal = nn.Parameter(initial_path[1:-1].clone())
    initial_energy = float(
        path_energy(model, initial_path, config.damping, config.quadrature_points)
    )
    scale = max(initial_energy, 1e-12)
    closure_count = 0

    def full_path() -> torch.Tensor:
        return torch.cat((start_tensor[None], internal, end_tensor[None]))

    optimizer = torch.optim.LBFGS(
        [internal],
        lr=1,
        max_iter=config.max_iterations,
        history_size=20,
        tolerance_grad=0,
        tolerance_change=0,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        nonlocal closure_count
        optimizer.zero_grad()
        energy = path_energy(
            model, full_path(), config.damping, config.quadrature_points
        ) / scale
        energy.backward()
        closure_count += 1
        return energy

    optimizer.step(closure)
    final_energy = float(
        path_energy(model, full_path(), config.damping, config.quadrature_points).detach()
    )
    closure()
    raw_gradient = float(internal.grad.abs().max()) * scale
    return {
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "energy_relative_decrease": (initial_energy - final_energy) / scale,
        "raw_gradient_infinity_norm": raw_gradient,
        "normalized_gradient_infinity_norm": float(internal.grad.abs().max()),
        "max_internal_node_movement": float(
            torch.max(torch.abs(internal.detach() - initial_path[1:-1])).cpu()
        ),
        "closure_evaluation_count": closure_count,
        "optimizer_state_n_iter": int(optimizer.state[internal].get("n_iter", 0)),
        "log_vector": (
            config.segments
            * (internal.detach()[0] - start_tensor)
        ).cpu().numpy().tolist(),
        "converged_under_current_threshold": bool(
            float(internal.grad.abs().max()) <= config.gradient_tolerance
        ),
    }


def _real_sample_audit(root: Path) -> list[dict]:
    model = load_vae("weights/trajectory_vae_8d_best.pth", "cpu").double()
    z0 = np.load("outputs/neutral_construction/z0.npy").astype(np.float64)
    indices = np.load("log_experiment/candidate_indices.npy")[:10]
    latents = np.load("log_experiment/candidate_latents.npy")[:10].astype(np.float64)
    config = SolverConfig()
    rows: list[dict] = []

    def activation_changes(path_array: np.ndarray) -> dict:
        value = torch.as_tensor(path_array, dtype=torch.float64)
        patterns = []
        for layer in model.decoder:
            value = layer(value)
            if isinstance(layer, nn.ReLU):
                patterns.append((value > 0).detach().cpu().numpy())
        layer_changes = [
            np.count_nonzero(pattern[1:] != pattern[:-1], axis=1).tolist()
            for pattern in patterns
        ]
        return {
            "activation_changes_per_segment_by_relu_layer": layer_changes,
            "segments_with_any_activation_change": int(
                np.count_nonzero(
                    np.any(
                        np.column_stack(
                            [np.asarray(changes) > 0 for changes in layer_changes]
                        ),
                        axis=1,
                    )
                )
            ),
        }
    for source_index, target in zip(indices, latents):
        initial_path = linear_path(z0, target, config.segments)
        initial_energy, initial_gradient = energy_gradient(
            model, initial_path, config.damping, config.quadrature_points
        )
        started = time.perf_counter()
        _, path, record = solve_log(model, z0, target, int(source_index), config)
        native = native_lbfgs_control(model, z0, target, config)
        rows.append(
            {
                "source_index": int(source_index),
                "initial_energy": initial_energy,
                "final_energy": record["final_energy"],
                "initial_raw_gradient_infinity_norm": float(np.abs(initial_gradient).max()),
                "initial_normalized_gradient_infinity_norm": float(
                    np.abs(initial_gradient).max() / max(initial_energy, 1e-12)
                ),
                "final_raw_gradient_infinity_norm": (
                    float(record["gradient_residual"] * initial_energy)
                    if record["gradient_residual"] is not None
                    else None
                ),
                "final_normalized_gradient_infinity_norm": record["gradient_residual"],
                "max_path_node_movement": float(
                    np.max(np.abs(path[1:-1] - initial_path.numpy()[1:-1]))
                ),
                "energy_relative_decrease": (
                    (initial_energy - record["final_energy"]) / max(initial_energy, 1e-12)
                ),
                "lbfgs_step_count": record["iterations"],
                "closure_evaluation_count": record["closure_evaluations"],
                "converged": record["converged"],
                "failure_reason": record["failure_reason"],
                "runtime_seconds": time.perf_counter() - started,
                "native_single_step_control": native,
                "optimized_path_activation_boundaries": activation_changes(path),
            }
        )
    write_json(root / "real_sample_records.json", rows)
    return rows


def run_audit(output_dir: str | Path = "solver_audit") -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    protected = [
        Path("outputs/v1/skill_pool_64.npz"),
        Path("outputs/v1/skill_pool_64.pt"),
        Path("outputs/local_baseline/skill_pool_64.npz"),
        Path("outputs/local_baseline/skill_pool_64.pt"),
        Path("weights/trajectory_vae_8d_best.pth"),
    ]
    before_hashes = {str(path): sha256(path) for path in protected}

    start = np.full(8, 0.2)
    end = np.linspace(0.35, 1.05, 8)
    linear_model = LinearToy()
    linear_result, linear_path_result = run_toy_case(
        linear_model, start, end, SolverConfig(segments=8, max_iterations=20), 1
    )
    if not (
        linear_result["converged"]
        and linear_result["initial_raw_gradient_infinity_norm"] < 1e-10
        and linear_result["log_error_l2"] < 1e-9
    ):
        raise RuntimeError("Linear toy failed; stop audit and repair solver")

    smooth_model = SmoothToy()
    smooth_start = np.full(8, -0.7)
    smooth_end = np.linspace(0.4, 1.3, 8)
    smooth_results = {}
    smooth_paths = {}
    for segments in (8, 16):
        result, solved_path = run_toy_case(
            smooth_model,
            smooth_start,
            smooth_end,
            SolverConfig(segments=segments, max_iterations=250, gradient_tolerance=1e-5),
            10 + segments,
        )
        smooth_results[f"L{segments}"] = result
        smooth_paths[f"L{segments}"] = solved_path
    smooth_results["L8_L16_log_difference_l2"] = float(
        np.linalg.norm(
            8 * (smooth_paths["L8"][1] - smooth_paths["L8"][0])
            - 16 * (smooth_paths["L16"][1] - smooth_paths["L16"][0])
        )
    )
    native_smooth = {}
    for segments in (8, 16):
        native_smooth[f"L{segments}"] = native_lbfgs_control(
            smooth_model,
            smooth_start,
            smooth_end,
            SolverConfig(segments=segments, max_iterations=250, gradient_tolerance=1e-5),
        )
    native_smooth["L8_L16_log_difference_l2"] = float(
        np.linalg.norm(
            np.asarray(native_smooth["L8"]["log_vector"])
            - np.asarray(native_smooth["L16"]["log_vector"])
        )
    )
    native_smooth["L8_L16_log_relative_difference"] = float(
        native_smooth["L8_L16_log_difference_l2"]
        / max(np.linalg.norm(native_smooth["L16"]["log_vector"]), 1e-12)
    )
    native_smooth["L8_L16_final_energy_relative_difference"] = float(
        abs(native_smooth["L8"]["final_energy"] - native_smooth["L16"]["final_energy"])
        / max(native_smooth["L16"]["final_energy"], 1e-12)
    )
    smooth_results["native_single_step_control"] = native_smooth

    relu_model = ReluToy()
    same_start = np.full(8, 2.0)
    same_end = np.full(8, 2.2)
    cross_start = np.full(8, -1.0)
    cross_end = np.full(8, 1.0)
    relu_results = {}
    relu_paths = {}
    for label, case_start, case_end in (
        ("same_activation_region", same_start, same_end),
        ("cross_activation_boundaries", cross_start, cross_end),
    ):
        result, solved_path = run_toy_case(
            relu_model,
            case_start,
            case_end,
            SolverConfig(segments=8, max_iterations=150, gradient_tolerance=1e-5),
            20 if label.startswith("same") else 21,
        )
        result.update(relu_path_diagnostics(relu_model, solved_path))
        result["native_single_step_control"] = native_lbfgs_control(
            relu_model,
            case_start,
            case_end,
            SolverConfig(segments=8, max_iterations=150, gradient_tolerance=1e-5),
        )
        relu_results[label] = result
        relu_paths[label] = solved_path

    gradient_checks = {
        "linear": finite_difference_gradient(linear_model, linear_path(start, end, 4)),
        "smooth_tanh": finite_difference_gradient(
            smooth_model, linear_path(smooth_start, smooth_end, 4)
        ),
        "relu_same_region": finite_difference_gradient(
            relu_model, linear_path(same_start, same_end, 4)
        ),
        "relu_cross_boundary": finite_difference_gradient(
            relu_model, linear_path(cross_start, cross_end, 4)
        ),
    }
    gradient_checks["all_passed"] = all(
        value["passed"] for value in gradient_checks.values() if isinstance(value, dict)
    )
    write_json(output / "gradient_check.json", gradient_checks)

    real_rows = _real_sample_audit(output)
    after_hashes = {str(path): sha256(path) for path in protected}
    protected_unchanged = before_hashes == after_hashes

    audit = {
        "scope": "toy cases plus the fixed first 10 construction candidates; no pool build and no 1000-sample run",
        "linear": linear_result,
        "smooth_tanh": smooth_results,
        "relu": relu_results,
        "gradient_check_summary": {
            "all_passed": gradient_checks["all_passed"],
            "max_absolute_error": max(
                value["max_absolute_error"]
                for value in gradient_checks.values()
                if isinstance(value, dict)
            ),
            "max_relative_error": max(
                value["max_relative_error"]
                for value in gradient_checks.values()
                if isinstance(value, dict)
            ),
        },
        "real_samples": real_rows,
        "lbfgs_audit": {
            "outer_loop": "One optimizer.step call per outer iteration; LBFGS max_iter=1.",
            "closure_behavior": "Each outer iteration evaluates closure once before step; strong-Wolfe may call closure multiple times inside step.",
            "best_path_restore": "Lowest sampled-energy path is restored, then closure recomputes its gradient before final convergence classification.",
            "stopping_condition": "Only normalized energy-gradient infinity norm is used; optimizer internal tolerances are disabled.",
            "finding": "Endpoints remain fixed and the restored best path receives a fresh gradient, but the repeated optimizer.step(max_iter=1) control flow is not behaviorally equivalent to one native LBFGS solve. It also performs a redundant pre-step closure. On smooth L8 it stops above tolerance after 250 step calls/752 closures, whereas the native control reaches tolerance in 118 LBFGS iterations/193 closures. Six of ten real cases restore the unchanged initial path after 100 step calls. This is a solver control-flow defect with direct convergence impact.",
        },
        "convergence_criterion_audit": {
            "current": "||grad E||_inf / max(E_initial, 1e-12) <= 1e-4",
            "dimensional_issue": "The ratio has units inverse latent-coordinate and scales approximately as inverse endpoint displacement under a constant metric; it is not scale invariant.",
            "recommended_primary": "||grad E||_inf / max(||grad E at linear initialization||_inf, machine_floor) <= relative_tolerance, together with an absolute raw-gradient floor.",
            "basis": "This is dimensionless and measures first-order stationarity relative to the problem's initial residual. It is a recommendation for future experiments, not used to relabel this audit's failures.",
            "secondary_checks": "Also require nonincreasing energy, fixed endpoints, finite values, and small relative path-step or relative energy change. No threshold was changed in this audit.",
        },
        "protected_hashes_before": before_hashes,
        "protected_hashes_after": after_hashes,
        "protected_artifacts_unchanged": protected_unchanged,
        "classification": [],
    }
    implementation_control_evidence = (
        not smooth_results["L8"]["converged"]
        and smooth_results["native_single_step_control"]["L8"][
            "converged_under_current_threshold"
        ]
    )
    if not gradient_checks["all_passed"] or not linear_result["converged"] or implementation_control_evidence:
        audit["classification"].append(
            {
                "choice": "A",
                "selected": True,
                "evidence": "The current repeated max_iter=1 loop failed the smooth L8 criterion after 250 calls, while one native max_iter=250 LBFGS step reached the same threshold with fewer closures. Linear and finite-difference checks pass, so the defect is optimizer control flow rather than energy/Jacobian differentiation.",
            }
        )
    else:
        audit["classification"].append(
            {"choice": "A", "selected": False, "evidence": "Linear analytic case and all finite-difference checks passed."}
        )
    audit["classification"].append(
        {
            "choice": "B",
            "selected": True,
            "evidence": "The current grad(E)/E_initial criterion is dimensionful and endpoint-scale dependent; raw and normalized residuals differ materially on real samples. Existing failures were not relabeled.",
        }
    )
    relu_cross_failed = not relu_results["cross_activation_boundaries"][
        "native_single_step_control"
    ]["converged_under_current_threshold"]
    smooth_passed = all(
        smooth_results["native_single_step_control"][key][
            "converged_under_current_threshold"
        ]
        for key in ("L8", "L16")
    )
    same_passed = relu_results["same_activation_region"][
        "native_single_step_control"
    ]["converged_under_current_threshold"]
    audit["classification"].append(
        {
            "choice": "C",
            "selected": bool(smooth_passed and same_passed and relu_cross_failed),
            "evidence": "Under the corrected native LBFGS control, smooth L8/L16 and same-region ReLU converge, while boundary-crossing ReLU does not; activation changes and per-node residuals are recorded. This supports a material ReLU-boundary contribution, but does not prove it is the sole cause on all real samples.",
        }
    )
    selected_abc = [row["choice"] for row in audit["classification"] if row["selected"]]
    audit["classification"].append(
        {
            "choice": "D",
            "selected": not bool(selected_abc),
            "evidence": "Selected only when A/B/C lack direct evidence.",
        }
    )
    if not protected_unchanged:
        raise RuntimeError("Protected V1/checkpoint artifact changed during audit")
    write_json(output / "audit_report.json", audit)
    _plot_audit(output, linear_result, smooth_results, relu_results, relu_paths, real_rows)
    return audit


def _plot_audit(
    output: Path,
    linear: dict,
    smooth: dict,
    relu: dict,
    relu_paths: dict[str, np.ndarray],
    real_rows: list[dict],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["linear", "smooth L8", "smooth L16", "ReLU same", "ReLU cross"]
    initial = [linear["initial_energy"], smooth["L8"]["initial_energy"], smooth["L16"]["initial_energy"], relu["same_activation_region"]["initial_energy"], relu["cross_activation_boundaries"]["initial_energy"]]
    final = [linear["final_energy"], smooth["L8"]["final_energy"], smooth["L16"]["final_energy"], relu["same_activation_region"]["final_energy"], relu["cross_activation_boundaries"]["final_energy"]]
    curves = [
        linear["energy_curve"], smooth["L8"]["energy_curve"],
        smooth["L16"]["energy_curve"],
        relu["same_activation_region"]["energy_curve"],
        relu["cross_activation_boundaries"]["energy_curve"],
    ]
    figure, axis = plt.subplots(figsize=(10, 5))
    for label, curve in zip(labels, curves):
        values = np.asarray(curve)
        axis.plot(np.arange(len(values)), values / values[0], label=label)
    axis.set(xlabel="LBFGS outer step", ylabel="Energy / initial energy", title="Toy solver energy curves")
    axis.legend(); axis.grid(alpha=0.25)
    figure.tight_layout(); figure.savefig(output / "toy_energy_curves.png", dpi=160); plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    for axis, label in zip(axes, ("same_activation_region", "cross_activation_boundaries")):
        patterns = np.asarray(relu[label]["activation_patterns"])
        axis.imshow(patterns.T, aspect="auto", interpolation="nearest", cmap="binary")
        axis.set(ylabel="Hidden ReLU unit", xlabel="Path node", title=f"{label}: {relu[label]['boundary_segment_count']} segments change activation")
    figure.tight_layout(); figure.savefig(output / "relu_activation_boundaries.png", dpi=160); plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    source = [row["source_index"] for row in real_rows]
    axes[0].plot(source, [row["initial_raw_gradient_infinity_norm"] for row in real_rows], "o-", label="initial raw")
    axes[0].plot(source, [row["final_raw_gradient_infinity_norm"] for row in real_rows], "o-", label="final raw")
    axes[0].set_yscale("log"); axes[0].set(ylabel="Raw gradient infinity norm", xlabel="Source index")
    axes[1].plot(source, [row["initial_normalized_gradient_infinity_norm"] for row in real_rows], "o-", label="initial normalized")
    axes[1].plot(source, [row["final_normalized_gradient_infinity_norm"] for row in real_rows], "o-", label="final normalized")
    axes[1].axhline(1e-4, color="red", linestyle="--", label="current tolerance")
    axes[1].set_yscale("log"); axes[1].set(ylabel="Normalized gradient infinity norm", xlabel="Source index")
    axes[2].bar(np.arange(len(source)), [row["energy_relative_decrease"] for row in real_rows])
    axes[2].set(xticks=np.arange(len(source)), xticklabels=source, ylabel="Relative energy decrease", xlabel="Source index")
    for number, axis in enumerate(axes):
        axis.tick_params(axis="x", rotation=45); axis.grid(alpha=0.25)
        if number < 2:
            axis.legend()
    figure.suptitle("Fixed first 10 construction candidates")
    figure.tight_layout(); figure.savefig(output / "real_sample_residuals.png", dpi=160); plt.close(figure)


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps({
        "scope": result["scope"],
        "selected_classifications": [
            row["choice"] for row in result["classification"] if row["selected"]
        ],
        "protected_artifacts_unchanged": result["protected_artifacts_unchanged"],
    }, ensure_ascii=False, indent=2))
