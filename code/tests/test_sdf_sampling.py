"""SDF sampling / analytic-shape sanity checks."""

import numpy as np
import trimesh

from src.data.synthetic import make_synthetic_collection
from src.data.sdf_sampling import sample_sdf_from_mesh
from src.data.normalize import make_watertight, normalize_mesh_to_unit_sphere


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


def _triangle_soup_box() -> trimesh.Trimesh:
    """A box mesh with every face given its own unshared vertices.

    Structurally like a raw ShapeNet OBJ: many disconnected shell fragments,
    not welded into one manifold surface (a real chair mesh checked directly
    had 313 disconnected bodies). ``make_watertight`` must handle this without
    caring whether ``signed_distance`` on the input happens to be reliable.
    """
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    verts = box.vertices[box.faces].reshape(-1, 3)  # (F*3, 3), no vertex sharing
    soup = trimesh.Trimesh(vertices=verts, faces=np.arange(len(verts)).reshape(-1, 3), process=False)
    assert not soup.is_watertight
    assert soup.body_count == len(box.faces)
    return soup


def test_make_watertight_repairs_fragmented_mesh():
    soup = normalize_mesh_to_unit_sphere(_triangle_soup_box())
    repaired = normalize_mesh_to_unit_sphere(make_watertight(soup, pitch=1.0 / 32.0))
    assert repaired.is_watertight
    assert repaired.body_count == 1

    # Repair should closely preserve the shape's volume, not just its topology
    # (a too-coarse pitch could balloon or shrink it enough to be useless).
    original_box = normalize_mesh_to_unit_sphere(trimesh.creation.box(extents=(1.0, 1.0, 1.0)))
    assert abs(repaired.volume - original_box.volume) / original_box.volume < 0.1

    # A watertight, single-body mesh gives a well-defined, non-trivial inside
    # fraction under signed_distance -- the actual training-label sign that a
    # fragmented mesh (verified directly on a real ShapeNet chair: 313 bodies)
    # makes close to meaningless (~1% "inside" there, vs. ~38% after repair).
    rng = np.random.default_rng(0)
    _, sdf_repaired = sample_sdf_from_mesh(repaired, num_points=1000, rng=rng)
    frac_repaired = (sdf_repaired < 0).mean()
    assert 0.15 < frac_repaired < 0.60
