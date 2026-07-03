"""Decode a latent code into an SDF volume and a mesh.

Pipeline: ``z -> f_theta(z, grid) -> SDF volume -> Marching Cubes -> mesh``.
Grid evaluation is chunked so high resolutions fit in memory on Colab.
"""

from __future__ import annotations

import numpy as np
import torch
import trimesh

from .data.sdf_sampling import sample_surface_points
from .marching import sdf_grid_to_mesh
from .models.decoder import DeepSDFDecoder


@torch.no_grad()
def decode_sdf_grid(
    decoder: DeepSDFDecoder,
    latent: torch.Tensor,
    resolution: int = 64,
    bound: float = 1.0,
    chunk: int = 65536,
    device: str = "cpu",
) -> np.ndarray:
    """Evaluate the decoder on a dense grid and return an SDF volume."""
    decoder.eval()
    latent = latent.to(device).view(1, -1)

    axis = torch.linspace(-bound, bound, resolution)
    grid = torch.stack(
        torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1
    ).reshape(-1, 3)

    values = torch.empty(grid.shape[0])
    for start in range(0, grid.shape[0], chunk):
        pts = grid[start : start + chunk].to(device)
        lat = latent.expand(pts.shape[0], -1)
        sdf = decoder(lat, pts).squeeze(-1).cpu()
        values[start : start + chunk] = sdf
    return values.reshape(resolution, resolution, resolution).numpy()


def decode_mesh(
    decoder: DeepSDFDecoder,
    latent: torch.Tensor,
    resolution: int = 64,
    bound: float = 1.0,
    device: str = "cpu",
) -> trimesh.Trimesh | None:
    """Decode a latent code straight to a mesh (or ``None`` if empty)."""
    volume = decode_sdf_grid(
        decoder, latent, resolution=resolution, bound=bound, device=device
    )
    return sdf_grid_to_mesh(volume, bound=bound, level=0.0)


def decode_point_cloud(
    decoder: DeepSDFDecoder,
    latent: torch.Tensor,
    num_points: int,
    resolution: int = 64,
    device: str = "cpu",
) -> np.ndarray | None:
    """Decode a latent code and sample a surface point cloud (for metrics)."""
    mesh = decode_mesh(decoder, latent, resolution=resolution, device=device)
    if mesh is None or len(mesh.faces) == 0:
        return None
    return sample_surface_points(mesh, num_points)
