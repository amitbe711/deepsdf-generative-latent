"""Reconstruction metrics: Chamfer distance and volumetric IoU."""

from __future__ import annotations

import warnings

import numpy as np
import trimesh
from scipy.spatial import cKDTree

_warned_no_embree = False


def _warn_if_slow_contains(resolution: int) -> None:
    """Flag the single-biggest footgun in this file: mesh.contains() without embree.

    Without the embreex package, trimesh's ray-mesh intersector falls back to a
    pure-Python ray-triangle test that is not just slow but effectively
    unbounded in memory -- verified directly: ~0.4s with embreex vs 200+s (and
    an OOM kernel crash on Colab) without it, for a single mesh at
    resolution=64. That failure mode is a silent hang/crash with no useful
    traceback, so warn loudly and immediately instead.
    """
    global _warned_no_embree
    if _warned_no_embree or trimesh.ray.has_embree or resolution < 48:
        return
    warnings.warn(
        f"embreex is not installed and iou_resolution={resolution} >= 48: "
        "mesh.contains() will use trimesh's pure-Python ray-triangle fallback, "
        "which is orders of magnitude slower and can exhaust memory (verified "
        "to OOM-crash a Colab kernel at resolution=64). Run `pip install "
        "embreex` (already in requirements.txt) before a real evaluation.",
        RuntimeWarning,
        stacklevel=2,
    )
    _warned_no_embree = True


def chamfer_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """Symmetric Chamfer distance (mean of squared nearest-neighbor distances).

    CD(A, B) = mean_a min_b ||a - b||^2 + mean_b min_a ||a - b||^2.
    """
    points_a = np.asarray(points_a, dtype=np.float32)
    points_b = np.asarray(points_b, dtype=np.float32)
    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)
    dist_a, _ = tree_b.query(points_a)  # a -> nearest in b
    dist_b, _ = tree_a.query(points_b)  # b -> nearest in a
    return float(np.mean(dist_a**2) + np.mean(dist_b**2))


def iou_from_meshes(
    mesh_pred: trimesh.Trimesh,
    mesh_gt: trimesh.Trimesh,
    resolution: int = 32,
    bound: float = 1.0,
) -> float:
    """Volumetric IoU by voxel occupancy over a shared grid.

    Occupancy is computed with mesh containment tests. Both meshes should be
    (approximately) watertight; Marching-Cubes outputs and analytic shapes are.
    """
    _warn_if_slow_contains(resolution)
    axis = np.linspace(-bound, bound, resolution, dtype=np.float32)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)

    occ_pred = mesh_pred.contains(grid)
    occ_gt = mesh_gt.contains(grid)
    intersection = np.logical_and(occ_pred, occ_gt).sum()
    union = np.logical_or(occ_pred, occ_gt).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)
