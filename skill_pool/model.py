from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn


class TrajectoryVAE(nn.Module):
    """Exact architecture used by real-Skillformer-master's 8-D checkpoint."""

    def __init__(
        self,
        latent_dim: int = 8,
        trajectory_length: int = 30,
        feature_dim: int = 2,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.trajectory_length = trajectory_length
        self.feature_dim = feature_dim
        input_dim = trajectory_length * feature_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, trajectory: Tensor) -> tuple[Tensor, Tensor]:
        encoded = self.encoder(trajectory.reshape(trajectory.shape[0], -1))
        return encoded.chunk(2, dim=-1)

    def decode(self, latent: Tensor) -> Tensor:
        decoded = self.decoder(latent)
        return decoded.reshape(
            latent.shape[0], self.trajectory_length, self.feature_dim
        )

    def forward(self, trajectory: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean, log_variance = self.encode(trajectory)
        standard_deviation = torch.exp(0.5 * log_variance)
        latent = mean + torch.randn_like(standard_deviation) * standard_deviation
        return self.decode(latent), mean, log_variance


def _load_state_dict(path: Path, device: torch.device) -> Mapping[str, Tensor]:
    try:
        value = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # torch < 2.0
        value = torch.load(path, map_location=device)
    if not isinstance(value, Mapping):
        raise TypeError(f"Checkpoint must contain a state_dict mapping: {path}")
    if "state_dict" in value and isinstance(value["state_dict"], Mapping):
        value = value["state_dict"]
    # Accept Lightning-style prefixes without weakening strict shape checking.
    prefixes = ("model.", "vae.", "model.vae.")
    keys = list(value)
    for prefix in prefixes:
        if keys and all(key.startswith(prefix) for key in keys):
            value = {key[len(prefix) :]: tensor for key, tensor in value.items()}
            break
    return value


def load_vae(
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
    latent_dim: int = 8,
    trajectory_length: int = 30,
) -> TrajectoryVAE:
    target_device = torch.device(device)
    model = TrajectoryVAE(
        latent_dim=latent_dim, trajectory_length=trajectory_length
    ).to(target_device)
    state_dict = _load_state_dict(Path(checkpoint), target_device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model

