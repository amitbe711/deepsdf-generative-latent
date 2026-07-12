"""Extract meshes from signed-distance fields via Marching Cubes.

Shared by the analytic synthetic shapes (``data/synthetic.py``) and the trained
decoder (``sample.py``). ``eval_sdf_on_grid`` evaluates any callable
``fn(points [M,3]) -> sdf [M]`` on a dense grid; ``sdf_grid_to_mesh`` runs
Marching Cubes (Lorensen & Cline, 1987) at the zero level set.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import trimesh
from skimage import measure


def eval_sdf_on_grid(
    fn: Callable[[np.ndarray], np.ndarray],
    resolution: int = 64,
    bound: float = 1.0,
    chunk: int = 65536,
) -> np.ndarray:
    """Evaluate an SDF callable on a ``resolution^3`` grid over ``[-bound, bound]^3``."""
    axis = np.linspace(-bound, bound, resolution, dtype=np.float32)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    points = grid.reshape(-1, 3)

    values = np.empty(points.shape[0], dtype=np.float32)
    for start in range(0, points.shape[0], chunk):
        end = start + chunk
        values[start:end] = fn(points[start:end])
    return values.reshape(resolution, resolution, resolution)


def sdf_grid_to_mesh(
    volume: np.ndarray, bound: float = 1.0, level: float = 0.0
) -> trimesh.Trimesh | None:
    """Marching Cubes on an SDF volume. Returns ``None`` if no surface is found."""
    vol = np.nan_to_num(volume, nan=0.0, posinf=1.0, neginf=-1.0)
    if vol.min() > level or vol.max() < level:
        # Under-trained / offset fields may not cross exactly 0; use the value
        # in the grid closest to zero (common with clamped-L1 + few real meshes).
        level = float(vol.flat[np.argmin(np.abs(vol))])
        if vol.min() == vol.max():
            return None
    resolution = vol.shape[0]
    spacing = (2.0 * bound) / (resolution - 1)
    try:
        verts, faces, normals, _ = measure.marching_cubes(
            vol, level=level, spacing=(spacing, spacing, spacing)
        )
    except (ValueError, RuntimeError):
        return None
    verts = verts - bound  # shift origin to the cube center
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)


def mesh_from_sdf_fn(
    fn: Callable[[np.ndarray], np.ndarray],
    resolution: int = 64,
    bound: float = 1.0,
    level: float = 0.0,
) -> trimesh.Trimesh | None:
    volume = eval_sdf_on_grid(fn, resolution=resolution, bound=bound)
    return sdf_grid_to_mesh(volume, bound=bound, level=level)
