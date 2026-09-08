from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .data import decode_latents, encode_trajectories, load_trajectories
from .evaluation import coverage_metrics, plot_pool, reconstruction_metrics, write_json
from .geometry import (
    choose_neutral_index,
    decoder_metric,
    select_direction_strength_pool,
)
from .model import load_vae


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_pipeline(
    trajectories_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    pool_sizes: tuple[int, ...] = (32, 64),
    strength_levels: int = 4,
    batch_size: int = 1024,
    device: str = "auto",
    seed: int = 7,
) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_device = resolve_device(device)

    trajectories = load_trajectories(trajectories_path)
    model = load_vae(checkpoint_path, selected_device)
    latents, log_variances = encode_trajectories(
        model, trajectories, batch_size, selected_device
    )
    np.save(output / "latents_mu.npy", latents)
    np.save(output / "latents_logvar.npy", log_variances)

    neutral_index = choose_neutral_index(trajectories)
    neutral_latent = latents[neutral_index]
    metric = decoder_metric(model, neutral_latent, selected_device)
    np.save(output / "decoder_metric_z0.npy", metric)
    np.save(output / "neutral_latent.npy", neutral_latent)

    # Reconstruction is a necessary checkpoint/data-coordinate compatibility check.
    reconstructions = decode_latents(model, latents, batch_size, selected_device)
    reconstruction = reconstruction_metrics(trajectories, reconstructions)

    report: dict = {
        "schema_version": 1,
        "latent_dim": 8,
        "trajectory_shape": [30, 2],
        "coordinate_convention": "+Y forward, X lateral, ego-local rear-axle origin",
        "selection_method": "decoder pullback metric + spherical FPS + strength quantiles",
        "is_clustering": False,
        "device": selected_device,
        "neutral_source_index": neutral_index,
        "reconstruction": reconstruction,
        "pools": {},
    }

    for pool_size in pool_sizes:
        pool, source_indices, direction_ids, strengths = select_direction_strength_pool(
            latents=latents,
            neutral_latent=neutral_latent,
            metric=metric,
            pool_size=pool_size,
            strength_levels=strength_levels,
        )
        decoded = decode_latents(model, pool, batch_size, selected_device)
        metrics = coverage_metrics(trajectories, decoded)
        stem = f"skill_pool_{pool_size}"
        np.savez_compressed(
            output / f"{stem}.npz",
            latents=pool,
            trajectories=decoded,
            source_indices=source_indices,
            direction_ids=direction_ids,
            strengths=strengths,
            neutral_latent=neutral_latent,
            decoder_metric=metric,
        )
        torch.save(
            {
                "schema_version": 1,
                "latent_dim": 8,
                "pool_size": pool_size,
                "latents": torch.from_numpy(pool),
                "trajectories": torch.from_numpy(decoded),
                "source_indices": torch.from_numpy(source_indices),
                "direction_ids": torch.from_numpy(direction_ids),
                "strengths": torch.from_numpy(strengths),
                "neutral_latent": torch.from_numpy(neutral_latent),
                "decoder_metric": torch.from_numpy(metric),
                "coordinate_convention": "+Y forward, X lateral",
            },
            output / f"{stem}.pt",
        )
        plot_pool(decoded, direction_ids, output / f"{stem}.png", f"Skill Pool K={pool_size}")
        report["pools"][str(pool_size)] = metrics

    write_json(output / "report.json", report)
    (output / "run_config.json").write_text(
        json.dumps(
            {
                "trajectories": str(Path(trajectories_path).resolve()),
                "checkpoint": str(Path(checkpoint_path).resolve()),
                "pool_sizes": list(pool_sizes),
                "strength_levels": strength_levels,
                "batch_size": batch_size,
                "device": selected_device,
                "seed": seed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return report

