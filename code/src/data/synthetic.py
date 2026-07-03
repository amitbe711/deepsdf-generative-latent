"""Analytic parametric shape family used for fast local development and as a
fallback when ShapeNet is unavailable.

Each shape exposes an *exact* analytic SDF (union of rounded boxes), so training
data needs no mesh->SDF preprocessing and ground-truth meshes are recovered with
Marching Cubes. The family is a simple parametric "chair" (seat + backrest +
four legs) whose proportions vary, giving a coherent shape *category* on which
generation metrics (coverage / MMD / 1-NN) are meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from ..marching import mesh_from_sdf_fn


def _box_sdf(points: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Exact signed distance to an axis-aligned box (negative inside)."""
    q = np.abs(points - center) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return (outside + inside).astype(np.float32)


class AnalyticShape:
    """Base class: a shape defined by a union (min) of analytic primitives."""

    def __init__(self, boxes: list[tuple[np.ndarray, np.ndarray]]) -> None:
        # Each entry is (center [3], half_extents [3]).
        self.boxes = boxes

    def sdf(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        dists = [_box_sdf(points, c, h) for c, h in self.boxes]
        return np.min(np.stack(dists, axis=0), axis=0)

    def to_mesh(self, resolution: int = 64) -> trimesh.Trimesh:
        mesh = mesh_from_sdf_fn(self.sdf, resolution=resolution, bound=1.0)
        if mesh is None:
            raise RuntimeError("Analytic shape produced no surface at level 0.")
        return mesh


@dataclass
class ChairParams:
    seat_w: float
    seat_d: float
    seat_h: float  # z of seat top
    seat_t: float  # seat slab thickness
    back_h: float  # backrest height above seat
    leg_t: float   # leg half-thickness


class ParametricChair(AnalyticShape):
    """A chair built from a seat slab, a backrest, and four legs."""

    def __init__(self, params: ChairParams) -> None:
        self.params = params
        p = params
        boxes: list[tuple[np.ndarray, np.ndarray]] = []

        seat_center = np.array([0.0, 0.0, p.seat_h], dtype=np.float32)
        seat_half = np.array([p.seat_w, p.seat_d, p.seat_t], dtype=np.float32)
        boxes.append((seat_center, seat_half))

        back_center = np.array(
            [0.0, -p.seat_d + p.leg_t, p.seat_h + p.back_h], dtype=np.float32
        )
        back_half = np.array([p.seat_w, p.leg_t, p.back_h], dtype=np.float32)
        boxes.append((back_center, back_half))

        leg_half = np.array([p.leg_t, p.leg_t, p.seat_h], dtype=np.float32)
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                cx = sx * (p.seat_w - p.leg_t)
                cy = sy * (p.seat_d - p.leg_t)
                boxes.append(
                    (np.array([cx, cy, 0.0], dtype=np.float32), leg_half)
                )
        super().__init__(boxes)


def make_synthetic_collection(
    num_shapes: int, seed: int = 0
) -> list[ParametricChair]:
    """Generate a reproducible family of parametric chairs with varied proportions."""
    rng = np.random.default_rng(seed)
    shapes: list[ParametricChair] = []
    for _ in range(num_shapes):
        params = ChairParams(
            seat_w=float(rng.uniform(0.30, 0.45)),
            seat_d=float(rng.uniform(0.30, 0.45)),
            seat_h=float(rng.uniform(0.20, 0.35)),
            seat_t=float(rng.uniform(0.04, 0.07)),
            back_h=float(rng.uniform(0.25, 0.45)),
            leg_t=float(rng.uniform(0.03, 0.05)),
        )
        shapes.append(ParametricChair(params))
    return shapes
