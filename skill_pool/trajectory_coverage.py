"""Construction-only trajectory-space farthest-point skill pools.

The selected trajectories cover the empirical future-trajectory support of the
current nuPlan mini extraction and frozen VAE.  They are not a claim about all
physically feasible driving trajectories.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import json

import numpy as np
import torch

from .data import decode_latents
from .model import load_vae
from .splits import aligned_arrays, sha256, write_json
from .v1 import verify_neutral


DT = 0.1
QUANTILES = (0.0, 0.5, 0.9, 0.95, 1.0)


def trajectory_descriptors(trajectories: np.ndarray, dt: float = DT) -> dict[str, np.ndarray]:
    trajectories = np.asarray(trajectories)
    origin = np.zeros((len(trajectories), 1, 2), dtype=trajectories.dtype)
    delta = np.diff(np.concatenate((origin, trajectories), axis=1), axis=1)
    speed = np.linalg.norm(delta, axis=-1) / dt
    moving = speed > 0.2
    heading = np.unwrap(np.arctan2(delta[..., 0], delta[..., 1]), axis=1)
    reliable_heading = np.where(moving, heading, np.nan)
    heading_change = np.asarray([
        np.ptp(row[np.isfinite(row)]) if np.isfinite(row).any() else 0.0
        for row in reliable_heading
    ])
    dh = np.diff(heading, axis=1)
    segment_distance = (
        np.linalg.norm(delta[:, 1:], axis=-1)
        + np.linalg.norm(delta[:, :-1], axis=-1)
    ) / 2
    reliable_curve = moving[:, 1:] & moving[:, :-1]
    curvature = np.where(
        reliable_curve, np.abs(dh) / np.maximum(segment_distance, 0.02), 0.0
    )
    return {
        "mean_speed_mps": speed.mean(axis=1),
        "speed_change_mps": speed[:, -5:].mean(axis=1) - speed[:, :5].mean(axis=1),
        "speed_std_mps": speed.std(axis=1),
        "max_step_speed_mps": speed.max(axis=1),
        "final_forward_m": trajectories[:, -1, 1],
        "final_lateral_m": trajectories[:, -1, 0],
        "max_lateral_m": np.abs(trajectories[..., 0]).max(axis=1),
        "max_abs_coordinate_m": np.abs(trajectories).max(axis=(1, 2)),
        "heading_change_deg": np.degrees(heading_change),
        "max_curvature_inv_m": curvature.max(axis=1),
    }


def filter_decoded_candidates(trajectories: np.ndarray) -> tuple[np.ndarray, dict, dict]:
    """Remove numerical/clearly implausible decoder output without pruning rare turns."""
    trajectories = np.asarray(trajectories)
    finite = np.isfinite(trajectories).all(axis=(1, 2))
    safe = np.nan_to_num(trajectories, nan=0.0, posinf=1e9, neginf=-1e9)
    descriptors = trajectory_descriptors(safe)
    # Broad physical sanity bounds.  Lateral displacement, heading change, and
    # curvature are deliberately not filters so valid rare/strong turns remain.
    reasons = {
        "nonfinite": ~finite,
        "coordinate_over_150m": descriptors["max_abs_coordinate_m"] > 150.0,
        "mean_speed_over_50mps": descriptors["mean_speed_mps"] > 50.0,
        "instantaneous_speed_over_80mps": descriptors["max_step_speed_mps"] > 80.0,
        "speed_std_over_30mps": descriptors["speed_std_mps"] > 30.0,
    }
    accepted = np.ones(len(trajectories), dtype=bool)
    assigned = np.full(len(trajectories), "accepted", dtype="<U40")
    for label, mask in reasons.items():
        newly_rejected = accepted & mask
        assigned[newly_rejected] = label
        accepted &= ~mask
    report = {
        "input_count": int(len(trajectories)),
        "accepted_count": int(accepted.sum()),
        "rejected_count": int((~accepted).sum()),
        "exclusive_reason_counts": dict(Counter(assigned[~accepted].tolist())),
        "overlapping_reason_counts": {key: int(mask.sum()) for key, mask in reasons.items()},
        "bounds": {
            "max_abs_coordinate_m": 150.0,
            "max_mean_speed_mps": 50.0,
            "max_instantaneous_speed_mps": 80.0,
            "max_speed_std_mps": 30.0,
        },
        "turn_preservation": "No lateral, heading-change, or curvature cutoff is applied.",
    }
    return accepted, report, descriptors


def robust_trajectory_scales(trajectories: np.ndarray) -> np.ndarray:
    """Construction-only per-time/per-coordinate IQR scale with stable floors."""
    q25, q75 = np.quantile(trajectories, [0.25, 0.75], axis=0)
    scale = (q75 - q25) / 1.349
    global_q25, global_q75 = np.quantile(trajectories, [0.25, 0.75], axis=(0, 1))
    global_scale = (global_q75 - global_q25) / 1.349
    floors = np.maximum(global_scale * 0.05, np.array([0.05, 0.10]))
    return np.maximum(scale, floors[None, :])


def trajectory_distance(
    candidates: np.ndarray,
    skill: np.ndarray,
    mode: str = "ade",
    scales: np.ndarray | None = None,
    endpoint_weight: float = 1.0,
) -> np.ndarray:
    difference = np.asarray(candidates) - np.asarray(skill)[None]
    if mode == "ade":
        point_distance = np.linalg.norm(difference, axis=-1)
    elif mode == "normalized":
        if scales is None or np.asarray(scales).shape != skill.shape:
            raise ValueError("Normalized distance requires one positive scale per trajectory coordinate")
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("Scales must be positive and finite")
        point_distance = np.linalg.norm(difference / scales[None], axis=-1)
    else:
        raise ValueError(f"Unknown trajectory distance mode: {mode}")
    weights = np.ones(skill.shape[0], dtype=np.float64)
    if not np.isfinite(endpoint_weight) or endpoint_weight <= 0:
        raise ValueError("Endpoint weight must be positive and finite")
    weights[-1] = endpoint_weight
    return np.average(point_distance, axis=1, weights=weights)


def trajectory_fps(
    trajectories: np.ndarray,
    source_indices: np.ndarray,
    neutral_source_index: int,
    pool_size: int,
    *,
    mode: str = "ade",
    scales: np.ndarray | None = None,
    endpoint_weight: float = 1.0,
    batch_size: int = 4096,
    seed: int = 7,
) -> tuple[np.ndarray, dict]:
    """Streaming FPS; memory is O(N), never O(N^2)."""
    trajectories = np.asarray(trajectories)
    source_indices = np.asarray(source_indices)
    if len(trajectories) != len(source_indices) or len(np.unique(source_indices)) != len(source_indices):
        raise ValueError("Candidate/source mapping mismatch")
    if not 1 <= pool_size <= len(trajectories):
        raise ValueError("Pool size outside candidate range")
    neutral_positions = np.flatnonzero(source_indices == neutral_source_index)
    if len(neutral_positions) != 1:
        raise ValueError("Neutral source must occur exactly once in candidates")
    rng = np.random.default_rng(seed)
    tie_rank = np.empty(len(trajectories), dtype=np.int64)
    tie_rank[rng.permutation(len(trajectories))] = np.arange(len(trajectories))
    selected = [int(neutral_positions[0])]
    minimum = np.full(len(trajectories), np.inf, dtype=np.float64)
    selection_distance = [0.0]
    max_block_rows = 0
    while len(selected) < pool_size:
        newest = trajectories[selected[-1]]
        for start in range(0, len(trajectories), batch_size):
            stop = min(start + batch_size, len(trajectories))
            distance = trajectory_distance(
                trajectories[start:stop], newest, mode, scales, endpoint_weight
            )
            minimum[start:stop] = np.minimum(minimum[start:stop], distance)
            max_block_rows = max(max_block_rows, stop - start)
        minimum[np.asarray(selected)] = -np.inf
        maximum = float(np.max(minimum))
        tied = np.flatnonzero(np.isclose(minimum, maximum, rtol=1e-12, atol=1e-14))
        choice = int(tied[np.argmin(tie_rank[tied])])
        selected.append(choice)
        selection_distance.append(maximum)
    diagnostics = {
        "distance_mode": mode,
        "endpoint_weight": float(endpoint_weight),
        "batch_size": int(batch_size),
        "largest_distance_vector_length": int(max_block_rows),
        "candidate_count": int(len(trajectories)),
        "full_pairwise_matrix_constructed": False,
        "seed": int(seed),
        "selection_source_indices": source_indices[selected].tolist(),
        "distance_at_selection": selection_distance,
    }
    return np.asarray(selected, dtype=np.int64), diagnostics


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values)
    if not len(values):
        return {key: None for key in ("min", "p50", "p90", "p95", "max", "mean")}
    names = ("min", "p50", "p90", "p95", "max")
    result = dict(zip(names, np.quantile(values, QUANTILES).tolist()))
    result["mean"] = float(values.mean())
    return result


def coverage_arrays(trajectories: np.ndarray, pool: np.ndarray, batch_size: int = 2048) -> dict:
    result = {key: [] for key in ("minADE_m", "minFDE_m", "FDE_at_minADE_m", "nearest_skill")}
    for start in range(0, len(trajectories), batch_size):
        batch = trajectories[start:start + batch_size]
        point = np.linalg.norm(batch[:, None] - pool[None], axis=-1)
        ade = point.mean(axis=-1)
        nearest = np.argmin(ade, axis=1)
        rows = np.arange(len(batch))
        result["minADE_m"].append(ade[rows, nearest])
        result["minFDE_m"].append(point[:, :, -1].min(axis=1))
        result["FDE_at_minADE_m"].append(point[rows, nearest, -1])
        result["nearest_skill"].append(nearest)
    return {key: np.concatenate(value) for key, value in result.items()}


def pairwise_summary(pool: np.ndarray) -> tuple[dict, np.ndarray]:
    matrix = np.linalg.norm(pool[:, None] - pool[None, :], axis=-1).mean(axis=-1)
    first, second = np.triu_indices(len(pool), 1)
    values = matrix[first, second]
    order = np.argsort(values, kind="stable")
    pairs = [
        {"skill_i": int(first[i]), "skill_j": int(second[i]), "ade_m": float(values[i])}
        for i in order[:10]
    ]
    return {
        "min_m": float(values.min()),
        "p10_m": float(np.quantile(values, 0.1)),
        "mean_m": float(values.mean()),
        "closest_pairs": pairs,
    }, matrix


def behavior_labels(descriptors: dict[str, np.ndarray]) -> np.ndarray:
    speed = descriptors["mean_speed_mps"]
    lateral = descriptors["final_lateral_m"]
    heading = descriptors["heading_change_deg"]
    label = np.full(len(speed), "straight", dtype="<U32")
    label[speed < 0.2] = "stationary"
    label[(speed >= 0.2) & (speed < 3)] = "slow"
    label[speed >= 8] = "fast"
    moving = speed >= 0.2
    turning = moving & (heading >= 10)
    label[turning & (lateral < 0)] = "turn_negative_X"
    label[turning & (lateral >= 0)] = "turn_positive_X"
    strong = turning & (heading >= 35)
    label[strong & (lateral < 0)] = "strong_turn_negative_X"
    label[strong & (lateral >= 0)] = "strong_turn_positive_X"
    return label


def behavior_breakdown(descriptors: dict[str, np.ndarray]) -> dict[str, dict[str, int]]:
    speed = descriptors["mean_speed_mps"]
    lateral = descriptors["final_lateral_m"]
    heading = descriptors["heading_change_deg"]
    speed_change = descriptors["speed_change_mps"]
    return {
        "speed": {
            "stationary_lt_0.2_mps": int(np.sum(speed < 0.2)),
            "slow_0.2_to_3_mps": int(np.sum((speed >= 0.2) & (speed < 3))),
            "medium_3_to_8_mps": int(np.sum((speed >= 3) & (speed < 8))),
            "fast_ge_8_mps": int(np.sum(speed >= 8)),
        },
        "lateral_direction": {
            "negative_X_lt_-0.5m": int(np.sum(lateral < -0.5)),
            "center_abs_le_0.5m": int(np.sum(np.abs(lateral) <= 0.5)),
            "positive_X_gt_0.5m": int(np.sum(lateral > 0.5)),
        },
        "turn_intensity": {
            "straight_lt_5deg": int(np.sum(heading < 5)),
            "mild_5_to_15deg": int(np.sum((heading >= 5) & (heading < 15))),
            "moderate_15_to_35deg": int(np.sum((heading >= 15) & (heading < 35))),
            "strong_ge_35deg": int(np.sum(heading >= 35)),
        },
        "speed_change": {
            "decelerating_lt_-1mps": int(np.sum(speed_change < -1)),
            "steady_abs_le_1mps": int(np.sum(np.abs(speed_change) <= 1)),
            "accelerating_gt_1mps": int(np.sum(speed_change > 1)),
        },
    }


def evaluate_pool(
    trajectories: np.ndarray,
    pool: np.ndarray,
    source_indices: np.ndarray,
    tokens: np.ndarray,
    scenario_types: np.ndarray,
    scenario_labels: list[str] | None = None,
) -> tuple[dict, dict]:
    samples = coverage_arrays(trajectories, pool)
    pairwise, _ = pairwise_summary(pool)
    counts = np.bincount(samples["nearest_skill"], minlength=len(pool))
    report = {
        "trajectory_count": int(len(trajectories)),
        "minADE_m": _summary(samples["minADE_m"]),
        "minFDE_m": _summary(samples["minFDE_m"]),
        "FDE_at_minADE_m": _summary(samples["FDE_at_minADE_m"]),
        "pool_pairwise_ADE": pairwise,
        "used_skill_count": int(np.count_nonzero(counts)),
        "assignment_counts": counts.tolist(),
        "by_scenario_type": {},
    }
    worst = int(np.argmax(samples["minADE_m"]))
    report["worst_trajectory"] = {
        "source_index": int(source_indices[worst]),
        "token": str(tokens[worst]),
        "scenario_type": str(scenario_types[worst]),
        "minADE_m": float(samples["minADE_m"][worst]),
        "nearest_skill": int(samples["nearest_skill"][worst]),
    }
    labels = scenario_labels if scenario_labels is not None else sorted(set(scenario_types.tolist()))
    for label in labels:
        mask = scenario_types == label
        count = int(mask.sum())
        report["by_scenario_type"][str(label)] = {
            "count": count,
            "minADE_m": _summary(samples["minADE_m"][mask]),
            "minFDE_m": _summary(samples["minFDE_m"][mask]),
            "FDE_at_minADE_m": _summary(samples["FDE_at_minADE_m"][mask]),
            "interpretation": (
                "no samples; no generalization conclusion" if count == 0 else
                "insufficient samples; descriptive only, no generalization conclusion" if count < 30 else
                "descriptive empirical coverage"
            ),
        }
    return report, samples


def _limits(candidate_trajectories: np.ndarray, pool128: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    x = np.concatenate((candidate_trajectories[..., 0].ravel(), pool128[..., 0].ravel(), [0]))
    y = np.concatenate((candidate_trajectories[..., 1].ravel(), pool128[..., 1].ravel(), [0]))
    xq = np.quantile(x, [0.001, 0.999]); yq = np.quantile(y, [0.001, 0.999])
    selected_x = pool128[..., 0]; selected_y = pool128[..., 1]
    xmin, xmax = min(xq[0], selected_x.min(), 0), max(xq[1], selected_x.max(), 0)
    ymin, ymax = min(yq[0], selected_y.min(), 0), max(yq[1], selected_y.max(), 0)
    xpad = max((xmax - xmin) * 0.05, 0.5); ypad = max((ymax - ymin) * 0.05, 0.5)
    return (float(xmin - xpad), float(xmax + xpad)), (float(ymin - ypad), float(ymax + ypad))


def plot_diagnostics(
    output: Path,
    pools: dict[int, np.ndarray],
    candidate_trajectories: np.ndarray,
    pool_reports: dict,
    pool_samples: dict,
    heldout_trajectories: np.ndarray,
    heldout_indices: np.ndarray,
    heldout_types: np.ndarray,
    neutral: np.ndarray,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xlim, ylim = _limits(candidate_trajectories, pools[128])
    candidate_end = candidate_trajectories[:, -1]
    for k, pool in pools.items():
        end = pool[:, -1]
        descriptors = trajectory_descriptors(pool)
        labels = behavior_labels(descriptors)
        colors = plt.get_cmap("viridis")(np.linspace(0, 1, k))
        for clean in (False, True):
            fig, ax = plt.subplots(figsize=(8, 9))
            for i, trajectory in enumerate(pool):
                ax.plot(trajectory[:, 0], trajectory[:, 1], color=colors[i], alpha=.75, lw=1.25)
                if not clean:
                    ax.annotate(str(i), trajectory[-1], fontsize=5)
            ax.plot(neutral[:, 0], neutral[:, 1], color="black", lw=3, label="skill 0 neutral")
            ax.scatter([0], [0], marker="x", color="red", s=55, label="ego origin", zorder=5)
            ax.set(xlim=xlim, ylim=ylim, xlabel="X lateral (m)", ylabel="Y forward (m)",
                   title=f"Trajectory-space coverage K={k}" + (" — clean fan" if clean else ""))
            ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=.2); ax.legend()
            fig.tight_layout(); fig.savefig(output / (f"skill_pool_clean_{k}.png" if clean else f"skill_pool_{k}.png"), dpi=170); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(end[1:, 0], end[1:, 1], c=np.arange(1, k), cmap="viridis", s=30)
        ax.scatter(end[0, 0], end[0, 1], marker="*", color="black", s=140, label="skill 0 neutral")
        ax.scatter([0], [0], marker="x", color="red", s=55, label="ego origin")
        ax.set(xlim=xlim, ylim=ylim, xlabel="X lateral endpoint (m)", ylabel="Y forward endpoint (m)", title=f"Skill endpoints K={k}")
        ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(output/f"skill_endpoints_{k}.png", dpi=170); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 8))
        density = ax.hexbin(candidate_end[:, 0], candidate_end[:, 1], gridsize=50, bins="log", mincnt=1, cmap="Greys")
        ax.scatter(end[1:, 0], end[1:, 1], color="tab:blue", s=24, label="selected endpoints")
        ax.scatter(end[0, 0], end[0, 1], marker="*", color="orange", edgecolor="black", s=150, label="skill 0 neutral")
        ax.scatter([0], [0], marker="x", color="red", s=55, label="ego origin")
        ax.set(xlim=xlim, ylim=ylim, xlabel="X lateral endpoint (m)", ylabel="Y forward endpoint (m)", title=f"Construction endpoint density + K={k}")
        ax.set_aspect("equal", adjustable="box"); ax.legend(); fig.colorbar(density, ax=ax, label="log bin count")
        fig.tight_layout(); fig.savefig(output/f"endpoint_density_{k}.png", dpi=170); plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 6))
        forward_bins = np.quantile(candidate_end[:, 1], np.linspace(0, 1, 9))
        lateral_bins = np.quantile(candidate_end[:, 0], np.linspace(0, 1, 9))
        occupancy, _, _ = np.histogram2d(end[:, 1], end[:, 0], bins=(forward_bins, lateral_bins))
        image = ax.imshow(occupancy, origin="lower", aspect="auto", cmap="Blues")
        ax.set(xlabel="Lateral endpoint construction-quantile bin", ylabel="Forward endpoint construction-quantile bin", title=f"Endpoint quantile-cell coverage K={k}")
        fig.colorbar(image, ax=ax, label="selected skill count"); fig.tight_layout(); fig.savefig(output/f"lateral_forward_coverage_{k}.png", dpi=170); plt.close(fig)

        breakdown = behavior_breakdown(descriptors)
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for ax, (dimension, counts) in zip(axes.ravel(), breakdown.items()):
            names = list(counts); ax.bar(names, list(counts.values()))
            ax.set(ylabel="Skill count", title=dimension.replace("_", " "))
            ax.tick_params(axis="x", rotation=25, labelsize=8); ax.grid(axis="y", alpha=.2)
        fig.suptitle(f"Speed, lateral direction, turn and speed-change coverage K={k}")
        fig.tight_layout(); fig.savefig(output/f"behavior_counts_{k}.png", dpi=170); plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        for ax, split in zip(axes, ("construction", "heldout")):
            counts = pool_reports[str(k)][split]["assignment_counts"]
            ax.bar(np.arange(k), counts); ax.set(ylabel="Assignments", title=split); ax.grid(axis="y", alpha=.2)
        axes[-1].set_xlabel("Skill ID"); fig.suptitle(f"Nearest-skill usage K={k}")
        fig.tight_layout(); fig.savefig(output/f"skill_usage_{k}.png", dpi=170); plt.close(fig)

        held = pool_samples[k]["heldout"]
        order = np.argsort(held["minADE_m"], kind="stable")
        positions = [0, len(order)//2, int(.9*(len(order)-1)), int(.95*(len(order)-1)), len(order)-1]
        fig, axes = plt.subplots(1, 5, figsize=(23, 7))
        for ax, position, title in zip(axes, positions, ("Best", "Median", "P90", "P95", "Worst")):
            local = int(order[position]); skill = int(held["nearest_skill"][local])
            ax.plot(heldout_trajectories[local, :, 0], heldout_trajectories[local, :, 1], label="held-out", lw=2)
            ax.plot(pool[skill, :, 0], pool[skill, :, 1], label=f"skill {skill}", lw=2)
            ax.scatter([0], [0], marker="x", color="red")
            ax.set(xlim=xlim, ylim=ylim, xlabel="X lateral (m)", ylabel="Y forward (m)",
                   title=f"{title}\nrow {heldout_indices[local]}, ADE {held['minADE_m'][local]:.2f} m\n{heldout_types[local]}")
            ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=.2); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(output/f"nearest_skill_examples_{k}.png", dpi=170); plt.close(fig)

        pairs = pool_reports[str(k)]["pairwise"]["closest_pairs"][:5]
        fig, axes = plt.subplots(1, 5, figsize=(20, 6))
        for ax, pair in zip(axes, pairs):
            for skill in (pair["skill_i"], pair["skill_j"]):
                ax.plot(pool[skill, :, 0], pool[skill, :, 1], label=f"skill {skill}")
            ax.scatter([0], [0], marker="x", color="red")
            ax.set(xlim=xlim, ylim=ylim, xlabel="X lateral (m)", ylabel="Y forward (m)", title=f"ADE {pair['ade_m']:.2f} m")
            ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=.2); ax.legend(fontsize=7)
        fig.suptitle(f"Five closest skill pairs K={k}"); fig.tight_layout(); fig.savefig(output/f"closest_skill_pairs_{k}.png", dpi=170); plt.close(fig)


def _comparison_plot(output: Path, report: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = ("mean", "p90", "p95", "max")
    fig, axes = plt.subplots(1, 4, figsize=(17, 5))
    for ax, metric in zip(axes, metrics):
        for method, style in (("trajectory_space", "o-"), ("local_baseline", "s--")):
            values, ks = [], []
            for k in (32, 64, 128):
                row = report["comparison"].get(str(k), {}).get(method)
                if row is not None:
                    ks.append(k); values.append(row["heldout"]["minADE_m"][metric])
            ax.plot(ks, values, style, label=method)
        ax.set(xlabel="Pool size K", ylabel="Held-out minADE (m)", title=metric); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output/"coverage_comparison.png", dpi=170); plt.close(fig)


def _smoke_plot(output: Path, pool: np.ndarray, candidates: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for trajectory in pool:
        axes[0].plot(trajectory[:, 0], trajectory[:, 1], alpha=.65)
    axes[0].plot(pool[0, :, 0], pool[0, :, 1], color="black", lw=3, label="skill 0 neutral")
    axes[0].scatter([0], [0], marker="x", color="red", label="ego origin")
    axes[0].set(xlabel="X lateral (m)", ylabel="Y forward (m)", title="1,000-candidate smoke fan")
    axes[0].set_aspect("equal", adjustable="datalim"); axes[0].legend(); axes[0].grid(alpha=.2)
    axes[1].hexbin(candidates[:, -1, 0], candidates[:, -1, 1], gridsize=35, bins="log", mincnt=1, cmap="Greys")
    axes[1].scatter(pool[1:, -1, 0], pool[1:, -1, 1], s=20, label="selected endpoints")
    axes[1].scatter(pool[0, -1, 0], pool[0, -1, 1], marker="*", color="orange", edgecolor="black", s=140, label="skill 0 neutral")
    axes[1].scatter([0], [0], marker="x", color="red", label="ego origin")
    axes[1].set(xlabel="X lateral endpoint (m)", ylabel="Y forward endpoint (m)", title="Candidate density and selected endpoints")
    axes[1].set_aspect("equal", adjustable="datalim"); axes[1].legend(); axes[1].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/"smoke_1000.png", dpi=170); plt.close(fig)


def run_pipeline(
    output_dir: str | Path = "trajectory_coverage_pool",
    candidate_limit: int | None = None,
    max_k: int = 128,
    endpoint_weight: float = 2.0,
    seed: int = 7,
    make_plots: bool = True,
) -> dict:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    raw, tokens, _, scenario_types = aligned_arrays("data/ego_trajs.npy", "data/ego_trajs_provenance.json")
    construction = np.load("splits/construction_indices.npy")
    heldout = np.load("splits/heldout_indices.npy")
    latents = np.load("outputs/v1/construction_latents.npy")
    latent_indices = np.load("outputs/v1/construction_indices.npy")
    if not np.array_equal(construction, latent_indices):
        raise ValueError("Construction latent/source mapping differs from split")
    z0, _, neutral_source = verify_neutral(
        "outputs/neutral_construction", construction, raw, "weights/trajectory_vae_8d_best.pth"
    )
    model = load_vae("weights/trajectory_vae_8d_best.pth", "cpu")
    decoded = decode_latents(model, latents, 1024, "cpu")
    all_scenario_labels = sorted(set(scenario_types.tolist()))
    accepted, filter_report, all_descriptors = filter_decoded_candidates(decoded)
    if not accepted[np.flatnonzero(construction == neutral_source)[0]]:
        raise ValueError("Validated neutral was rejected by candidate filter")
    candidate_positions = np.flatnonzero(accepted)
    if candidate_limit is not None:
        if candidate_limit < max_k:
            raise ValueError("Candidate limit must be at least max_k")
        rng = np.random.default_rng(seed)
        neutral_position = int(np.flatnonzero(construction == neutral_source)[0])
        others = candidate_positions[candidate_positions != neutral_position]
        candidate_positions = np.concatenate(([neutral_position], rng.choice(others, candidate_limit - 1, replace=False)))
    candidate_latents = latents[candidate_positions]
    candidate_indices = construction[candidate_positions]
    candidate_trajectories = decoded[candidate_positions]
    scales = robust_trajectory_scales(candidate_trajectories)
    sequences, fps_reports = {}, {}
    for mode in ("ade", "normalized"):
        sequence, diagnostics = trajectory_fps(
            candidate_trajectories, candidate_indices, neutral_source, max_k,
            mode=mode, scales=scales if mode == "normalized" else None,
            endpoint_weight=endpoint_weight if mode == "normalized" else 1.0,
            seed=seed,
        )
        sequences[mode] = sequence; fps_reports[mode] = diagnostics
    # The method is fixed before held-out evaluation.  The construction-only
    # 1,000-candidate smoke test selected ADE because it had lower mean/tail
    # physical-space ADE and covered stationary/straight/turning behaviors.
    selected = sequences["ade"]
    neutral_decoded = np.load("outputs/neutral_construction/neutral_decoded.npy")
    if not np.array_equal(candidate_latents[selected[0]], z0):
        raise ValueError("Skill 0 latent differs from validated construction z0")
    if not np.array_equal(candidate_trajectories[selected[0]], neutral_decoded):
        raise ValueError("Skill 0 decoded trajectory differs from validated neutral artifact")
    neutral_token = str(np.load("outputs/neutral_construction/neutral_source_token.npy"))
    if int(candidate_indices[selected[0]]) != neutral_source or str(tokens[neutral_source]) != neutral_token:
        raise ValueError("Skill 0 source/token differs from validated neutral artifact")

    np.save(output/"candidate_source_indices.npy", candidate_indices)
    np.save(output/"candidate_latents.npy", candidate_latents)
    np.save(output/"candidate_decoded_trajectories.npy", candidate_trajectories)
    np.save(output/"trajectory_scales.npy", scales)
    np.save(output/"selection_sequence_ade.npy", candidate_indices[sequences["ade"]])
    np.save(output/"selection_sequence_normalized.npy", candidate_indices[sequences["normalized"]])
    np.savez_compressed(
        output/"candidate_descriptors.npz", source_indices=candidate_indices,
        **{key: value[candidate_positions] for key, value in all_descriptors.items()},
    )
    write_json(output/"candidate_filter_report.json", {
        **filter_report, "candidate_limit": candidate_limit,
        "descriptor_file": "candidate_descriptors.npz",
        "descriptor_fields": list(all_descriptors),
    })
    write_json(output/"fps_selection_report.json", {
        "selection_split": "construction", "selected_method": "ade", "methods": fps_reports,
        "selection_rationale": "Chosen before held-out evaluation from the 1,000-candidate construction smoke test: lower construction physical-space mean/tail minADE and broader stationary/straight/speed representation than the normalized variant.",
        "normalized_distance": "weighted mean over all 30 points of Euclidean coordinate residual divided by construction-only per-time/per-axis robust scale",
        "ade_distance": "mean Euclidean distance over all 30 trajectory points",
        "scale_method": "per-time/per-axis IQR/1.349 with construction-derived 5% global-scale floors and fixed numerical floors X=.05m,Y=.10m",
        "endpoint_weight": endpoint_weight, "trajectory_scales": scales.tolist(),
    })

    ks = [k for k in (32, 64, 128) if k <= max_k]
    evaluation_splits = [("construction", construction)]
    if candidate_limit is None:
        # Held-out is first touched here, after distance definition, scales, and
        # both nested construction-only selection sequences are frozen.
        evaluation_splits.append(("heldout", heldout))
    pools, reports, samples = {}, {}, {}
    for k in ks:
        positions = selected[:k]; pool = candidate_trajectories[positions]
        pools[k] = pool; samples[k] = {}
        pool_descriptors = trajectory_descriptors(pool)
        pairwise, _ = pairwise_summary(pool)
        reports[str(k)] = {
            "selection_split": "construction", "nested_prefix_of_K128": bool(np.array_equal(positions, selected[:k])),
            "source_indices": candidate_indices[positions].tolist(), "pairwise": pairwise,
            "behavior_counts": dict(Counter(behavior_labels(pool_descriptors).tolist())),
            "behavior_breakdown": behavior_breakdown(pool_descriptors),
        }
        payload = {
            "latents": candidate_latents[positions], "trajectories": pool,
            "source_indices": candidate_indices[positions], "source_tokens": tokens[candidate_indices[positions]],
            "source_trajectories": raw[candidate_indices[positions]], "skill_ids": np.arange(k),
            "selection_distances": np.asarray(fps_reports["ade"]["distance_at_selection"][:k]),
            "selection_split": np.array("construction"), "distance_mode": np.array("ade"),
            "endpoint_weight": np.array(1.0), "seed": np.array(seed),
        }
        np.savez_compressed(output/f"skill_pool_{k}.npz", **payload)
        torch.save({key: (torch.from_numpy(value) if isinstance(value, np.ndarray) and value.dtype.kind not in "US" else value.tolist()) for key, value in payload.items()}, output/f"skill_pool_{k}.pt")
        for split, indices in evaluation_splits:
            split_report, split_samples = evaluate_pool(
                raw[indices], pool, indices, tokens[indices], scenario_types[indices], all_scenario_labels
            )
            reports[str(k)][split] = split_report; samples[k][split] = split_samples
            np.savez_compressed(output/f"coverage_{split}_{k}.npz", source_indices=indices, **split_samples)

    # Distance A is compared without using held-out to select either sequence.
    distance_comparison = {}
    for mode, sequence in sequences.items():
        distance_comparison[mode] = {}
        for k in ks:
            pool = candidate_trajectories[sequence[:k]]
            distance_comparison[mode][str(k)] = {
                split: evaluate_pool(raw[indices], pool, indices, tokens[indices], scenario_types[indices], all_scenario_labels)[0]
                for split, indices in evaluation_splits
            }

    local_comparison = {}
    if candidate_limit is None:
        for k in (32, 64):
            local_pool = np.load(f"outputs/local_baseline/skill_pool_{k}.npz")["trajectories"]
            local_comparison[str(k)] = {
                split: evaluate_pool(raw[indices], local_pool, indices, tokens[indices], scenario_types[indices], all_scenario_labels)[0]
                for split, indices in evaluation_splits
            }
    comparison = {}
    for k in ks:
        comparison[str(k)] = {"trajectory_space": {split: reports[str(k)][split] for split, _ in evaluation_splits}}
        if str(k) in local_comparison:
            comparison[str(k)]["local_baseline"] = local_comparison[str(k)]

    report = {
        "method": "trajectory-space coverage pool",
        "scope_claim": "Coverage of the empirical feasible trajectory space supported by the current nuPlan mini extraction and frozen VAE; not all theoretically physically feasible trajectories.",
        "selection_split": "construction", "evaluation_splits": [name for name, _ in evaluation_splits],
        "seed": seed, "candidate_limit": candidate_limit, "construction_count": int(len(construction)),
        "heldout_count": int(len(heldout)), "candidate_count_after_filter": int(len(candidate_indices)),
        "neutral_source_index": int(neutral_source), "neutral_token": neutral_token,
        "neutral_exact_match": {"latent": True, "source_index": True, "token": True, "decoded_trajectory": True},
        "filter": filter_report, "distance_comparison": distance_comparison, "pools": reports,
        "comparison": comparison,
        "protected_inputs": {
            "local_baseline_report_sha256": sha256("outputs/local_baseline/report.json"),
            "v1_report_sha256": sha256("outputs/v1/report.json"),
            "neutral_check_sha256": sha256("outputs/neutral_construction/neutral_check.json"),
            "checkpoint_sha256": sha256("weights/trajectory_vae_8d_best.pth"),
            "construction_indices_sha256": sha256("splits/construction_indices.npy"),
            "heldout_indices_sha256": sha256("splits/heldout_indices.npy"),
        },
    }
    write_json(output/"report.json", report)
    if make_plots and max_k == 128:
        if candidate_limit is None:
            plot_diagnostics(output, pools, candidate_trajectories, reports, samples,
                             raw[heldout], heldout, scenario_types[heldout], neutral_decoded)
            _comparison_plot(output, report)
        else:
            _smoke_plot(output, pools[128], candidate_trajectories)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="trajectory_coverage_pool")
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--max-k", type=int, default=128)
    parser.add_argument("--endpoint-weight", type=float, default=2.0)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    report = run_pipeline(args.output_dir, args.candidate_limit, args.max_k, args.endpoint_weight, make_plots=not args.no_plots)
    print(json.dumps({key: report[key] for key in ("method", "selection_split", "candidate_count_after_filter")}, indent=2))


if __name__ == "__main__":
    main()
