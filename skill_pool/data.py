from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .model import TrajectoryVAE


def load_trajectories(path: str | Path, trajectory_length: int = 30) -> np.ndarray:
    trajectories = np.load(path, allow_pickle=False)
    if trajectories.ndim != 3 or trajectories.shape[1:] != (trajectory_length, 2):
        raise ValueError(
            f"Expected [N,{trajectory_length},2], got {trajectories.shape} from {path}"
        )
    trajectories = np.asarray(trajectories, dtype=np.float32)
    valid = np.isfinite(trajectories).all(axis=(1, 2))
    trajectories = trajectories[valid]
    if len(trajectories) == 0:
        raise ValueError("No finite trajectories remain after validation")
    return trajectories


def encode_trajectories(
    model: TrajectoryVAE,
    trajectories: np.ndarray,
    batch_size: int,
    device: str | torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    means: list[np.ndarray] = []
    log_variances: list[np.ndarray] = []
    target_device = torch.device(device)
    with torch.inference_mode():
        for start in range(0, len(trajectories), batch_size):
            batch = torch.from_numpy(trajectories[start : start + batch_size]).to(
                target_device
            )
            mean, log_variance = model.encode(batch)
            means.append(mean.cpu().numpy())
            log_variances.append(log_variance.cpu().numpy())
    return np.concatenate(means), np.concatenate(log_variances)


def decode_latents(
    model: TrajectoryVAE,
    latents: np.ndarray,
    batch_size: int,
    device: str | torch.device,
) -> np.ndarray:
    decoded: list[np.ndarray] = []
    target_device = torch.device(device)
    with torch.inference_mode():
        for start in range(0, len(latents), batch_size):
            batch = torch.from_numpy(
                np.asarray(latents[start : start + batch_size], dtype=np.float32)
            ).to(target_device)
            decoded.append(model.decode(batch).cpu().numpy())
    return np.concatenate(decoded)

