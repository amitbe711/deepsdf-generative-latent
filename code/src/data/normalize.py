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


def make_watertight(mesh: trimesh.Trimesh, pitch: float = 1.0 / 64.0) -> trimesh.Trimesh:
    """Repair a mesh into a single watertight, manifold solid via voxel remeshing.

    Raw ShapeNet OBJs are typically *not* watertight: a chair is usually dozens
    to hundreds of disconnected shell fragments (seat, each leg, back, cushions
    ...) with no single well-defined "inside". ``trimesh.proximity.signed_distance``
    (ray/winding-number based) silently produces near-meaningless signs on such
    meshes -- verified empirically: ~1% of sampled points came back "inside" for
    a typical raw chair mesh (313 disconnected bodies), vs. ~38% after this
    repair, for the same sample locations. Since the sign is the actual DeepSDF
    training label, this is a training-*data* bug that no amount of decoder/
    optimizer tuning on top can fix -- it explains low point-wise loss
    (the network dutifully fits whatever label it's given) coexisting with
    fragmented/incoherent extracted meshes, and why *more capacity made it
    worse* (fits the wrong labels more precisely).

    Voxelizing, filling the interior, and re-meshing with marching cubes always
    yields a single closed, watertight solid regardless of how broken the input
    topology was, at the cost of losing detail thinner than ``pitch``. Call this
    on an already-normalized (unit-sphere) mesh so ``pitch`` is in comparable
    units across shapes; re-normalize the output afterward since voxelization
    shifts the bounding box slightly.
    """
    voxels = mesh.voxelized(pitch=pitch).fill()
    return voxels.marching_cubes
