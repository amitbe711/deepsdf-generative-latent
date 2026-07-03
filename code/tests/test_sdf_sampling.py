"""SDF sampling / analytic-shape sanity checks."""

import numpy as np

from src.data.synthetic import make_synthetic_collection
from src.data.sdf_sampling import sample_sdf_from_mesh


def test_analytic_sdf_sign_convention():
    shape = make_synthetic_collection(1, seed=0)[0]
    # A point in the middle of the seat slab must be inside (negative).
    seat_center = np.array([[0.0, 0.0, shape.params.seat_h]], dtype=np.float32)
    assert shape.sdf(seat_center)[0] < 0.0
    # A far corner of the cube is well outside (positive).
    far = np.array([[0.95, 0.95, 0.95]], dtype=np.float32)
    assert shape.sdf(far)[0] > 0.0


def test_analytic_shape_meshes():
    shape = make_synthetic_collection(1, seed=1)[0]
    mesh = shape.to_mesh(resolution=32)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
    # Vertices live inside the unit cube.
    assert np.abs(mesh.vertices).max() <= 1.01


def test_mesh_to_sdf_sign():
    shape = make_synthetic_collection(1, seed=2)[0]
    mesh = shape.to_mesh(resolution=48)
    points, sdf = sample_sdf_from_mesh(mesh, num_points=2000, rng=np.random.default_rng(0))
    assert points.shape == (2000, 3)
    assert sdf.shape == (2000,)
    # Both interior (negative) and exterior (positive) samples must appear.
    assert (sdf < 0).any()
    assert (sdf > 0).any()
