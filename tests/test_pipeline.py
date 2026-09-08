from pathlib import Path

import numpy as np
import torch

from skill_pool.model import TrajectoryVAE
from skill_pool.pipeline import run_pipeline


def _synthetic_trajectories(count: int = 160) -> np.ndarray:
    rng = np.random.default_rng(11)
    time = np.linspace(0.0, 3.0, 30, dtype=np.float32)
    trajectories = []
    for _ in range(count):
        speed = rng.uniform(1.0, 12.0)
        lateral = rng.uniform(-7.0, 7.0)
        curve = rng.uniform(-1.0, 1.0)
        x = lateral * (time / 3.0) + curve * (time / 3.0) ** 2
        y = speed * time + rng.uniform(-0.4, 0.4) * time**2
        trajectories.append(np.stack([x, y], axis=-1))
    return np.asarray(trajectories, dtype=np.float32)


def test_pipeline_writes_a_complete_pool(tmp_path: Path):
    trajectories = _synthetic_trajectories()
    trajectory_path = tmp_path / "trajectories.npy"
    checkpoint_path = tmp_path / "vae.pth"
    np.save(trajectory_path, trajectories)
    torch.save(TrajectoryVAE().state_dict(), checkpoint_path)

    report = run_pipeline(
        trajectory_path,
        checkpoint_path,
        tmp_path / "output",
        pool_sizes=(8,),
        strength_levels=2,
        batch_size=64,
        device="cpu",
    )

    pool = np.load(tmp_path / "output" / "skill_pool_8.npz")
    assert pool["latents"].shape == (8, 8)
    assert pool["trajectories"].shape == (8, 30, 2)
    assert len(set(pool["source_indices"].tolist())) == 8
    assert report["pools"]["8"]["pool_size"] == 8
    for name in ("skill_pool_8.pt", "skill_pool_8.png", "report.json"):
        assert (tmp_path / "output" / name).exists()
