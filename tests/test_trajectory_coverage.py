from pathlib import Path
import json

import numpy as np

from skill_pool.trajectory_coverage import (
    filter_decoded_candidates,
    robust_trajectory_scales,
    trajectory_distance,
    trajectory_fps,
)


ROOT = Path(__file__).resolve().parents[1]


def _fan() -> tuple[np.ndarray, np.ndarray]:
    time = np.linspace(0.1, 3.0, 30)
    trajectories = [np.column_stack((np.zeros(30), 4 * time))]
    for lateral, forward in [(-8, 12), (8, 12), (0, 3), (0, 25), (-3, 20), (3, 20)]:
        fraction = time / time[-1]
        trajectories.append(np.column_stack((lateral * fraction**2, forward * fraction)))
    return np.asarray(trajectories), np.arange(100, 107)


def test_streaming_fps_selects_current_farthest_and_is_reproducible():
    trajectories, indices = _fan()
    scales = robust_trajectory_scales(trajectories)
    first, audit = trajectory_fps(trajectories, indices, 100, 7, mode="normalized", scales=scales, endpoint_weight=2, batch_size=3, seed=7)
    second, _ = trajectory_fps(trajectories, indices, 100, 7, mode="normalized", scales=scales, endpoint_weight=2, batch_size=2, seed=7)
    np.testing.assert_array_equal(first, second)
    chosen = [first[0]]
    for actual in first[1:]:
        minimum = np.min(np.column_stack([
            trajectory_distance(trajectories, trajectories[item], "normalized", scales, 2)
            for item in chosen
        ]), axis=1)
        minimum[chosen] = -np.inf
        assert minimum[actual] == np.max(minimum)
        chosen.append(actual)
    assert audit["largest_distance_vector_length"] <= 3
    assert not audit["full_pairwise_matrix_constructed"]


def test_synthetic_fan_covers_sides_straight_and_distances():
    trajectories, indices = _fan()
    selected, _ = trajectory_fps(trajectories, indices, 100, 7, mode="ade", batch_size=3)
    endpoints = trajectories[selected, -1]
    assert endpoints[:, 0].min() <= -8 and endpoints[:, 0].max() >= 8
    assert np.any(endpoints[:, 0] == 0)
    assert endpoints[:, 1].min() <= 3 and endpoints[:, 1].max() >= 25


def test_duplicate_straights_do_not_fill_available_pool_slots():
    fan, _ = _fan()
    duplicated = np.concatenate((fan, np.repeat(fan[:1], 100, axis=0)))
    indices = np.arange(len(duplicated))
    selected, _ = trajectory_fps(duplicated, indices, 0, 7, mode="ade", batch_size=11)
    assert np.sum(np.all(duplicated[selected] == fan[0], axis=(1, 2))) == 1


def test_candidate_filter_rejects_numerical_extremes_but_preserves_strong_turns():
    fan, _ = _fan()
    candidates = fan[[0, 1]].copy()
    nonfinite = fan[0].copy(); nonfinite[4, 0] = np.nan
    extreme = fan[0].copy(); extreme[-1, 1] = 200
    candidates = np.concatenate((candidates, nonfinite[None], extreme[None]))
    accepted, report, descriptors = filter_decoded_candidates(candidates)
    np.testing.assert_array_equal(accepted, [True, True, False, False])
    assert descriptors["heading_change_deg"][1] > 0
    assert report["exclusive_reason_counts"] == {"nonfinite": 1, "coordinate_over_150m": 1}


def test_formal_pools_are_nested_construction_only_and_decode_consistent():
    from skill_pool.data import decode_latents
    from skill_pool.model import load_vae
    output = ROOT / "trajectory_coverage_pool"
    construction = np.load(ROOT / "splits/construction_indices.npy")
    heldout = np.load(ROOT / "splits/heldout_indices.npy")
    neutral_z0 = np.load(ROOT / "outputs/neutral_construction/z0.npy")
    neutral_decoded = np.load(ROOT / "outputs/neutral_construction/neutral_decoded.npy")
    pools = {k: np.load(output/f"skill_pool_{k}.npz") for k in (32, 64, 128)}
    np.testing.assert_array_equal(pools[32]["source_indices"], pools[64]["source_indices"][:32])
    np.testing.assert_array_equal(pools[64]["source_indices"], pools[128]["source_indices"][:64])
    for pool in pools.values():
        assert np.isin(pool["source_indices"], construction).all()
        assert not np.isin(pool["source_indices"], heldout).any()
        np.testing.assert_array_equal(pool["latents"][0], neutral_z0)
        np.testing.assert_array_equal(pool["trajectories"][0], neutral_decoded)
        assert int(pool["source_indices"][0]) == int(np.load(ROOT/"outputs/neutral_construction/neutral_source_index.npy"))
        candidates = np.load(output/"candidate_source_indices.npy")
        decoded = np.load(output/"candidate_decoded_trajectories.npy")
        lookup = {int(source): i for i, source in enumerate(candidates)}
        np.testing.assert_array_equal(pool["trajectories"], decoded[[lookup[int(i)] for i in pool["source_indices"]]])
    model = load_vae(ROOT/"weights/trajectory_vae_8d_best.pth", "cpu")
    fresh = decode_latents(model, pools[128]["latents"], 128, "cpu")
    np.testing.assert_array_equal(fresh, pools[128]["trajectories"])


def test_selection_is_construction_only_and_heldout_independent():
    report = json.loads((ROOT/"trajectory_coverage_pool/report.json").read_text())
    selection = json.loads((ROOT/"trajectory_coverage_pool/fps_selection_report.json").read_text())
    assert report["selection_split"] == "construction"
    assert selection["selection_split"] == "construction"
    assert "heldout" not in json.dumps(selection).lower()
    before = np.load(ROOT/"trajectory_coverage_pool/selection_sequence_ade.npy")
    # Selection function has no held-out argument; unrelated held-out contents cannot enter it.
    candidates = np.load(ROOT/"trajectory_coverage_pool/candidate_decoded_trajectories.npy")[:200]
    sources = np.load(ROOT/"trajectory_coverage_pool/candidate_source_indices.npy")[:200]
    neutral = int(sources[0])
    scales = robust_trajectory_scales(candidates)
    a, _ = trajectory_fps(candidates, sources, neutral, 32, mode="normalized", scales=scales, endpoint_weight=2)
    unrelated_heldout = np.full((10, 30, 2), 1e9)
    unrelated_heldout *= -3
    b, _ = trajectory_fps(candidates, sources, neutral, 32, mode="normalized", scales=scales, endpoint_weight=2)
    np.testing.assert_array_equal(a, b)
    assert len(before) == 128
