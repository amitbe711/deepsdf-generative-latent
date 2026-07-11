"""Build shape collections and expose them as SDF training tensors.

A "shape collection" is a list of per-shape dicts::

    {"points": [P,3], "sdf": [P], "surface": [S,3], "mesh": trimesh | None}

Two sources are supported:
  * ``synthetic`` - analytic parametric chairs (exact SDF, no preprocessing).
  * ``mesh_dir`` - a directory of meshes (e.g. a ShapeNet chairs subset), sampled
    with the trimesh proximity query.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

from .normalize import normalize_mesh_to_unit_sphere
from .sdf_sampling import sample_sdf_from_mesh, sample_surface_points
from .synthetic import AnalyticShape, make_synthetic_collection
from ..utils.log import status

_MESH_EXTENSIONS = (".obj", ".off", ".ply", ".stl", ".glb")


def _samples_from_analytic(
    shape: AnalyticShape,
    num_points: int,
    surface_ratio: float,
    sigmas: tuple[float, float],
    num_surface_pc: int,
    mesh_resolution: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    mesh = shape.to_mesh(resolution=mesh_resolution)
    num_surface = int(round(num_points * surface_ratio))
    num_uniform = num_points - num_surface

    surf = sample_surface_points(mesh, num_surface)
    half = num_surface // 2
    noise = np.empty_like(surf)
    noise[:half] = rng.normal(0.0, sigmas[0], size=(half, 3))
    noise[half:] = rng.normal(0.0, sigmas[1], size=(num_surface - half, 3))
    near = surf + noise.astype(np.float32)
    uniform = rng.uniform(-1.0, 1.0, size=(num_uniform, 3)).astype(np.float32)

    points = np.concatenate([near, uniform], axis=0).astype(np.float32)
    sdf = shape.sdf(points).astype(np.float32)  # exact analytic SDF
    surface = sample_surface_points(mesh, num_surface_pc)
    return {"points": points, "sdf": sdf, "surface": surface, "mesh": mesh}


def _samples_from_mesh(
    mesh: trimesh.Trimesh,
    num_points: int,
    surface_ratio: float,
    sigmas: tuple[float, float],
    num_surface_pc: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    mesh = normalize_mesh_to_unit_sphere(mesh)
    points, sdf = sample_sdf_from_mesh(
        mesh,
        num_points=num_points,
        surface_ratio=surface_ratio,
        sigmas=sigmas,
        rng=rng,
    )
    surface = sample_surface_points(mesh, num_surface_pc)
    return {"points": points, "sdf": sdf, "surface": surface, "mesh": mesh}


def _load_meshes_from_dir(path: Path, limit: int | None) -> list[trimesh.Trimesh]:
    """Load triangle meshes from a directory tree.

    ShapeNetCore v2 layout is supported: ``03001627/<model_id>/models/model_normalized.obj``.
    We prefer ``model_normalized.obj`` (one per model folder) and skip macOS junk.
    """
    def _is_junk(p: Path) -> bool:
        return "__MACOSX" in p.parts or p.name.startswith("._")

    normalized = sorted(
        p for p in path.rglob("model_normalized.obj") if not _is_junk(p)
    )
    if normalized:
        files = normalized
    else:
        files = sorted(
            p
            for p in path.rglob("*")
            if p.suffix.lower() in _MESH_EXTENSIONS and not _is_junk(p)
        )
    if limit is not None:
        files = files[:limit]

    meshes: list[trimesh.Trimesh] = []
    for file in files:
        try:
            loaded = trimesh.load(file, force="mesh", process=False)
        except Exception:
            continue
        if isinstance(loaded, trimesh.Trimesh) and len(loaded.faces) > 0:
            meshes.append(loaded)
    return meshes


def build_shape_collection(
    cfg: Any,
    num_shapes: int,
    offset: int = 0,
    verbose: bool = False,
    prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Construct ``num_shapes`` preprocessed shapes according to ``cfg.data``.

    For ``mesh_dir``, ``offset`` skips the first ``offset`` meshes so a held-out
    reference set can be disjoint from the training set.
    """
    data_cfg = cfg.data
    rng = np.random.default_rng(cfg.seed)
    sigmas = tuple(data_cfg.sigmas)
    surface_ratio = float(data_cfg.surface_ratio)
    num_points = int(data_cfg.num_points)
    num_surface_pc = int(data_cfg.num_surface_points)

    source = data_cfg.source
    collection: list[dict[str, Any]] = []

    if source == "synthetic":
        shapes = make_synthetic_collection(num_shapes, seed=cfg.seed)
        for shape in shapes:
            collection.append(
                _samples_from_analytic(
                    shape,
                    num_points=num_points,
                    surface_ratio=surface_ratio,
                    sigmas=sigmas,
                    num_surface_pc=num_surface_pc,
                    mesh_resolution=int(data_cfg.mesh_resolution),
                    rng=rng,
                )
            )
    elif source == "mesh_dir":
        meshes = _load_meshes_from_dir(Path(data_cfg.mesh_dir), limit=None)
        end = offset + num_shapes
        if len(meshes) < end:
            raise ValueError(
                f"Requested meshes [{offset}:{end}) but only found {len(meshes)} "
                f"in {data_cfg.mesh_dir}."
            )
        meshes = meshes[offset:end]
        if verbose:
            status(
                f"SDF-sampling {len(meshes)} meshes from {data_cfg.mesh_dir} "
                f"({num_points} pts/shape; ~2-5 min/mesh from Drive)",
                prefix=prefix,
            )
        import time as _time

        for i, mesh in enumerate(meshes):
            t_mesh = _time.time()
            collection.append(
                _samples_from_mesh(
                    mesh,
                    num_points=num_points,
                    surface_ratio=surface_ratio,
                    sigmas=sigmas,
                    num_surface_pc=num_surface_pc,
                    rng=rng,
                )
            )
            if verbose:
                status(
                    f"mesh {i + 1}/{len(meshes)} sampled ({_time.time() - t_mesh:.1f}s)",
                    prefix=prefix,
                )
    else:
        raise ValueError(f"Unknown data source: {source!r}")

    return collection


class ShapeSDFDataset:
    """Holds SDF samples for a collection of shapes as contiguous tensors."""

    def __init__(self, collection: list[dict[str, Any]]) -> None:
        self.collection = collection
        self.num_shapes = len(collection)
        self.points = torch.from_numpy(
            np.stack([s["points"] for s in collection], axis=0)
        ).float()  # (N, P, 3)
        self.sdf = torch.from_numpy(
            np.stack([s["sdf"] for s in collection], axis=0)
        ).float()  # (N, P)
        self.points_per_shape = self.points.shape[1]

    def random_batch(
        self,
        num_shapes: int,
        points_per_shape: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a training batch.

        Returns:
            shape_idx: (B,) long, points: (B, 3), sdf: (B,) with
            B = num_shapes * points_per_shape.
        """
        num_shapes = min(num_shapes, self.num_shapes)
        shape_ids = torch.randperm(self.num_shapes, generator=generator)[:num_shapes]

        pts_batch = []
        sdf_batch = []
        idx_batch = []
        for sid in shape_ids.tolist():
            point_ids = torch.randint(
                0, self.points_per_shape, (points_per_shape,), generator=generator
            )
            pts_batch.append(self.points[sid, point_ids])
            sdf_batch.append(self.sdf[sid, point_ids])
            idx_batch.append(torch.full((points_per_shape,), sid, dtype=torch.long))

        return (
            torch.cat(idx_batch, dim=0),
            torch.cat(pts_batch, dim=0),
            torch.cat(sdf_batch, dim=0),
        )

    def surface_clouds(self) -> list[np.ndarray]:
        return [s["surface"] for s in self.collection]

    def meshes(self) -> list[trimesh.Trimesh | None]:
        return [s.get("mesh") for s in self.collection]
