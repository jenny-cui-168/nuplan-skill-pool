from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "skill-pool-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def coverage_metrics(
    trajectories: np.ndarray,
    pool_trajectories: np.ndarray,
    chunk_size: int = 4096,
) -> dict[str, float | int]:
    minimum_ade: list[np.ndarray] = []
    minimum_fde: list[np.ndarray] = []
    ade_selected_fde: list[np.ndarray] = []
    nearest_ids: list[np.ndarray] = []
    for start in range(0, len(trajectories), chunk_size):
        batch = trajectories[start : start + chunk_size]
        distance = np.linalg.norm(
            batch[:, None, :, :] - pool_trajectories[None, :, :, :], axis=-1
        )
        ade = distance.mean(axis=-1)
        nearest = np.argmin(ade, axis=1)
        minimum_ade.append(ade[np.arange(len(batch)), nearest])
        minimum_fde.append(distance[:, :, -1].min(axis=1))
        ade_selected_fde.append(distance[np.arange(len(batch)), nearest, -1])
        nearest_ids.append(nearest)
    min_ade = np.concatenate(minimum_ade)
    min_fde = np.concatenate(minimum_fde)
    nearest = np.concatenate(nearest_ids)

    pairwise = np.linalg.norm(
        pool_trajectories[:, None] - pool_trajectories[None, :], axis=-1
    ).mean(axis=-1)
    off_diagonal = pairwise[~np.eye(len(pool_trajectories), dtype=bool)]
    counts = np.bincount(nearest, minlength=len(pool_trajectories))
    probabilities = counts[counts > 0] / counts.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    normalized_entropy = entropy / np.log(len(pool_trajectories)) if len(pool_trajectories) > 1 else 0.0
    return {
        "trajectory_count": int(len(trajectories)),
        "pool_size": int(len(pool_trajectories)),
        "coverage_minADE_mean_m": float(np.mean(min_ade)),
        "coverage_minADE_p90_m": float(np.quantile(min_ade, 0.90)),
        "coverage_minFDE_mean_m": float(np.mean(min_fde)),
        "coverage_FDE_at_minADE_mean_m": float(np.mean(np.concatenate(ade_selected_fde))),
        "pool_pairwise_ADE_mean_m": float(np.mean(off_diagonal)),
        "pool_pairwise_ADE_min_m": float(np.min(off_diagonal)),
        "used_skill_count": int(np.count_nonzero(counts)),
        "assignment_entropy_normalized": float(normalized_entropy),
    }


def reconstruction_metrics(
    trajectories: np.ndarray, reconstructions: np.ndarray
) -> dict[str, float]:
    distance = np.linalg.norm(trajectories - reconstructions, axis=-1)
    return {
        "vae_reconstruction_ADE_mean_m": float(np.mean(distance)),
        "vae_reconstruction_ADE_p90_m": float(np.quantile(distance.mean(axis=1), 0.90)),
        "vae_reconstruction_FDE_mean_m": float(np.mean(distance[:, -1])),
    }


def plot_pool(
    pool_trajectories: np.ndarray,
    direction_ids: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 9), constrained_layout=True)
    colors = plt.get_cmap("tab20")
    direction_count = max(int(direction_ids.max()) + 1, 1)
    for index, (trajectory, direction_id) in enumerate(
        zip(pool_trajectories, direction_ids)
    ):
        color = colors(direction_id % 20)
        axis.plot(trajectory[:, 0], trajectory[:, 1], color=color, alpha=0.85, linewidth=1.6)
        axis.scatter(trajectory[-1, 0], trajectory[-1, 1], color=color, s=14)
        axis.annotate(str(index), trajectory[-1], fontsize=6, alpha=0.8)
    axis.scatter([0], [0], color="black", marker="x", s=45, label="ego origin")
    axis.set_xlabel("lateral X (m)")
    axis.set_ylabel("forward Y (m)")
    axis.set_title(title)
    axis.axis("equal")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_json(path: str | Path, value: dict) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
