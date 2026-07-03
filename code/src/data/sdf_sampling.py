"""Sample signed-distance supervision from a triangle mesh.

Sign convention (DeepSDF): SDF < 0 inside the surface, SDF > 0 outside.
``trimesh.proximity.signed_distance`` returns *positive inside*, so we negate it.

Sampling strategy (DeepSDF, Sec. 3): most samples are drawn near the surface
(surface points perturbed by two Gaussians), plus a smaller fraction drawn
uniformly in the cube to constrain the far field.
"""

from __future__ import annotations

import numpy as np
import trimesh


def sample_surface_points(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
    """Uniformly sample ``num_points`` points on the mesh surface."""
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return np.asarray(points, dtype=np.float32)


def _signed_distance(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    """Signed distance with the DeepSDF sign convention (negative inside)."""
    query = trimesh.proximity.ProximityQuery(mesh)
    sd = query.signed_distance(points)  # positive inside (trimesh convention)
    return (-np.asarray(sd, dtype=np.float32)).astype(np.float32)


def sample_sdf_from_mesh(
    mesh: trimesh.Trimesh,
    num_points: int = 30000,
    surface_ratio: float = 0.9,
    sigmas: tuple[float, float] = (0.005, 0.02),
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(points [P,3], sdf [P])`` sampled from a normalized mesh.

    Args:
        mesh: mesh already normalized into the unit sphere.
        num_points: total number of samples.
        surface_ratio: fraction drawn near the surface (rest uniform in cube).
        sigmas: two perturbation std-devs applied to surface samples.
        rng: optional NumPy random generator for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng()

    num_surface = int(round(num_points * surface_ratio))
    num_uniform = num_points - num_surface

    # Near-surface: half perturbed by sigma[0], half by sigma[1].
    half = num_surface // 2
    surf = sample_surface_points(mesh, num_surface)
    noise = np.empty_like(surf)
    noise[:half] = rng.normal(0.0, sigmas[0], size=(half, 3))
    noise[half:] = rng.normal(0.0, sigmas[1], size=(num_surface - half, 3))
    near = surf + noise.astype(np.float32)

    # Uniform samples in the [-1, 1]^3 cube.
    uniform = rng.uniform(-1.0, 1.0, size=(num_uniform, 3)).astype(np.float32)

    points = np.concatenate([near, uniform], axis=0).astype(np.float32)
    sdf = _signed_distance(mesh, points)
    return points, sdf
