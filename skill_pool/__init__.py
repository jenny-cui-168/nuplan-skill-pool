"""Trajectory-VAE skill pool extraction."""

__all__ = ["TrajectoryVAE", "load_vae"]


def __getattr__(name):
    # Keep NumPy-only geometry utilities importable while a PyTorch environment
    # is still being installed.
    if name in __all__:
        from .model import TrajectoryVAE, load_vae

        return {"TrajectoryVAE": TrajectoryVAE, "load_vae": load_vae}[name]
    raise AttributeError(name)
