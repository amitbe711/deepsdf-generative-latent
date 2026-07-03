"""Reconstruction metrics: Chamfer distance and volumetric IoU."""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree


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
    axis = np.linspace(-bound, bound, resolution, dtype=np.float32)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)

    occ_pred = mesh_pred.contains(grid)
    occ_gt = mesh_gt.contains(grid)
    intersection = np.logical_and(occ_pred, occ_gt).sum()
    union = np.logical_or(occ_pred, occ_gt).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)
