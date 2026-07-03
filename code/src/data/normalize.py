"""Normalize meshes into a canonical unit sphere, following the DeepSDF convention.

DeepSDF trains on shapes normalized so that the surface lies inside the unit
sphere (a small margin is kept so that near-surface samples never leave the
[-1, 1]^3 cube used at inference time).
"""

from __future__ import annotations

import numpy as np
import trimesh


def normalize_mesh_to_unit_sphere(
    mesh: trimesh.Trimesh, buffer: float = 1.03
) -> trimesh.Trimesh:
    """Center a mesh at its bounding-box center and scale it into the unit sphere.

    Args:
        mesh: input triangle mesh.
        buffer: shrink factor > 1 leaving a small margin between the surface and
            the unit sphere (DeepSDF uses ~1.03).

    Returns:
        A copy of the mesh centered at the origin with max radius ``1 / buffer``.
    """
    mesh = mesh.copy()
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    mesh.apply_translation(-center)
    radius = float(np.linalg.norm(mesh.vertices, axis=1).max())
    if radius > 0:
        mesh.apply_scale(1.0 / (radius * buffer))
    return mesh
