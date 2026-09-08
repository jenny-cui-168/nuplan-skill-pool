from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from .model import TrajectoryVAE


def trajectory_descriptors(trajectories: np.ndarray, dt: float = 0.1) -> dict[str, np.ndarray]:
    """Descriptors for the real-Skillformer convention: +Y forward, X lateral."""
    delta = np.diff(trajectories, axis=1)
    speed = np.linalg.norm(delta, axis=-1) / dt
    headings = np.unwrap(np.arctan2(delta[..., 0], delta[..., 1]), axis=1)
    heading_change = np.mean(np.abs(np.diff(headings, axis=1)), axis=1)
    return {
        "forward_progress": trajectories[:, -1, 1] - trajectories[:, 0, 1],
        "lateral_displacement": trajectories[:, -1, 0] - trajectories[:, 0, 0],
        "lateral_excursion": np.max(
            np.abs(trajectories[..., 0] - trajectories[:, :1, 0]), axis=1
        ),
        "mean_speed": np.mean(speed, axis=1),
        "speed_variation": np.std(speed, axis=1),
        "heading_change": heading_change,
    }


def choose_neutral_index(trajectories: np.ndarray) -> int:
    """Choose a typical moving, nearly straight trajectory as the neutral skill."""
    desc = trajectory_descriptors(trajectories)
    moving = desc["forward_progress"] >= np.quantile(desc["forward_progress"], 0.25)
    straight_threshold = np.quantile(desc["lateral_excursion"], 0.35)
    candidates = np.flatnonzero(moving & (desc["lateral_excursion"] <= straight_threshold))
    if len(candidates) == 0:
        candidates = np.arange(len(trajectories))

    def robust_scale(values: np.ndarray) -> tuple[float, float]:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return median, max(1.4826 * mad, 1e-6)

    speed_center, speed_scale = robust_scale(desc["mean_speed"][candidates])
    lateral_center, lateral_scale = robust_scale(desc["lateral_excursion"][candidates])
    heading_center, heading_scale = robust_scale(desc["heading_change"][candidates])
    score = (
        np.abs(desc["mean_speed"][candidates] - speed_center) / speed_scale
        + np.abs(desc["lateral_excursion"][candidates] - lateral_center) / lateral_scale
        + np.abs(desc["heading_change"][candidates] - heading_center) / heading_scale
    )
    return int(candidates[np.argmin(score)])


def decoder_metric(
    model: "TrajectoryVAE",
    neutral_latent: np.ndarray,
    device: str | "torch.device",
    damping: float = 1e-4,
) -> np.ndarray:
    """Pullback metric G=J_D(z0)^T J_D(z0)/output_dim + damping*I."""
    import torch

    target_device = torch.device(device)
    latent = torch.as_tensor(neutral_latent, dtype=torch.float32, device=target_device)
    latent = latent.detach().requires_grad_(True)

    def flat_decode(value: torch.Tensor) -> torch.Tensor:
        return model.decode(value.unsqueeze(0)).reshape(-1)

    try:
        jacobian = torch.autograd.functional.jacobian(
            flat_decode, latent, create_graph=False, vectorize=True
        )
    except RuntimeError:
        jacobian = torch.autograd.functional.jacobian(
            flat_decode, latent, create_graph=False, vectorize=False
        )
    metric = jacobian.T @ jacobian / jacobian.shape[0]
    metric = metric + damping * torch.eye(
        metric.shape[0], dtype=metric.dtype, device=metric.device
    )
    return metric.detach().cpu().numpy().astype(np.float64)


def metric_norm(vectors: np.ndarray, metric: np.ndarray) -> np.ndarray:
    squared = np.einsum("ni,ij,nj->n", vectors, metric, vectors)
    return np.sqrt(np.maximum(squared, 0.0))


def normalize_directions(
    deltas: np.ndarray, metric: np.ndarray, epsilon: float = 1e-8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    strength = metric_norm(deltas, metric)
    valid = strength > epsilon
    directions = deltas[valid] / strength[valid, None]
    return directions, strength[valid], valid


def metric_cosine(
    left: np.ndarray, right: np.ndarray, metric: np.ndarray
) -> np.ndarray:
    return np.clip(left @ metric @ right.T, -1.0, 1.0)


def spherical_fps(
    directions: np.ndarray, metric: np.ndarray, count: int
) -> np.ndarray:
    """Greedy farthest-point sampling on the decoder-metric unit sphere."""
    if count < 1 or count > len(directions):
        raise ValueError(f"count must be in [1,{len(directions)}], got {count}")
    mean_direction = directions.mean(axis=0, keepdims=True)
    mean_norm = metric_norm(mean_direction, metric)[0]
    if mean_norm > 1e-8:
        mean_direction /= mean_norm
        first = int(np.argmin(metric_cosine(directions, mean_direction, metric)[:, 0]))
    else:
        first = 0
    selected = [first]
    min_distance = 1.0 - metric_cosine(
        directions, directions[[first]], metric
    )[:, 0]
    min_distance[first] = -np.inf
    for _ in range(1, count):
        index = int(np.argmax(min_distance))
        selected.append(index)
        distance = 1.0 - metric_cosine(
            directions, directions[[index]], metric
        )[:, 0]
        min_distance = np.minimum(min_distance, distance)
        min_distance[selected] = -np.inf
    return np.asarray(selected, dtype=np.int64)


def select_direction_strength_pool(
    latents: np.ndarray,
    neutral_latent: np.ndarray,
    metric: np.ndarray,
    pool_size: int,
    strength_levels: int = 4,
    strength_quantiles: tuple[float, ...] | None = None,
    trim_quantile: float = 0.995,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select actual encoded samples using spherical FPS and per-direction strengths."""
    if pool_size % strength_levels != 0:
        raise ValueError("pool_size must be divisible by strength_levels")
    if strength_quantiles is None:
        strength_quantiles = tuple(
            float(value) for value in np.linspace(0.25, 0.90, strength_levels)
        )
    if len(strength_quantiles) != strength_levels:
        raise ValueError("strength_quantiles must have strength_levels entries")

    deltas = np.asarray(latents, dtype=np.float64) - neutral_latent
    directions, strengths, valid = normalize_directions(deltas, metric)
    source_indices = np.flatnonzero(valid)
    keep = strengths <= np.quantile(strengths, trim_quantile)
    directions, strengths, source_indices = (
        directions[keep],
        strengths[keep],
        source_indices[keep],
    )
    direction_count = pool_size // strength_levels
    if len(directions) < pool_size:
        raise ValueError(f"Need at least {pool_size} non-neutral candidates")

    direction_seed_indices = spherical_fps(directions, metric, direction_count)
    seeds = directions[direction_seed_indices]
    similarities = metric_cosine(directions, seeds, metric)
    assignment = np.argmax(similarities, axis=1)

    chosen: list[int] = []
    direction_ids: list[int] = []
    chosen_strengths: list[float] = []
    used: set[int] = set()
    for direction_id in range(direction_count):
        members = np.flatnonzero(assignment == direction_id)
        if len(members) == 0:
            members = np.asarray([direction_seed_indices[direction_id]])
        member_strength = strengths[members]
        for quantile in strength_quantiles:
            target = float(np.quantile(member_strength, quantile))
            ranked = members[np.argsort(np.abs(member_strength - target))]
            picked = next(
                (int(candidate) for candidate in ranked if int(source_indices[candidate]) not in used),
                None,
            )
            if picked is None:
                global_rank = np.argsort(
                    (1.0 - similarities[:, direction_id])
                    + 0.05 * np.abs(strengths - target) / max(target, 1e-8)
                )
                picked = next(
                    int(candidate)
                    for candidate in global_rank
                    if int(source_indices[candidate]) not in used
                )
            original_index = int(source_indices[picked])
            used.add(original_index)
            chosen.append(original_index)
            direction_ids.append(direction_id)
            chosen_strengths.append(float(strengths[picked]))

    chosen_array = np.asarray(chosen, dtype=np.int64)
    return (
        np.asarray(latents[chosen_array], dtype=np.float32),
        chosen_array,
        np.asarray(direction_ids, dtype=np.int64),
        np.asarray(chosen_strengths, dtype=np.float32),
    )
