"""Preflight checks before a ShapeNet grid run (Databricks / local).

Validates mesh loading, SDF sampling (no NaNs), rtree, and a decode smoke test.
Exit code 0 = OK, 1 = failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-dir", type=str, required=True)
    parser.add_argument("--code-dir", type=str, default=".")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    code_dir = Path(args.code_dir).resolve()
    sys.path.insert(0, str(code_dir))

    mesh_dir = Path(args.mesh_dir)
    objs = sorted(mesh_dir.rglob("model_normalized.obj"))
    print(f"model_normalized.obj count: {len(objs)}")
    if len(objs) == 0:
        print("FAIL: no meshes found")
        sys.exit(1)

    # rtree (required for fast / reliable trimesh proximity on many meshes)
    try:
        import rtree  # noqa: F401
    except ImportError:
        print("FAIL: rtree not installed — pip install rtree")
        sys.exit(1)

    from src.data.dataset import _load_meshes_from_dir
    from src.data.sdf_sampling import sample_sdf_from_mesh
    from src.models.decoder import DeepSDFDecoder
    from src.sample import decode_mesh

    meshes = _load_meshes_from_dir(mesh_dir, limit=3)
    print(f"loader returned {len(meshes)} meshes")
    if not meshes:
        print("FAIL: loader found 0 valid triangle meshes")
        sys.exit(1)

    rng = np.random.default_rng(0)
    for i, mesh in enumerate(meshes):
        pts, sdf = sample_sdf_from_mesh(mesh, num_points=2000, rng=rng)
        nan_frac = float(np.isnan(sdf).mean())
        print(
            f"  mesh {i}: verts={len(mesh.vertices)} faces={len(mesh.faces)} "
            f"sdf nan_frac={nan_frac:.3f} min={np.nanmin(sdf):.4f} max={np.nanmax(sdf):.4f}"
        )
        if nan_frac > 0.01:
            print("FAIL: SDF samples contain too many NaNs (check rtree / mesh integrity)")
            sys.exit(1)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARN: CUDA unavailable, using cpu for decode smoke test")
        device = "cpu"
    else:
        print(f"CUDA: {torch.cuda.is_available()}", end="")
        if torch.cuda.is_available():
            print(f" ({torch.cuda.get_device_name(0)})")
        else:
            print()

    decoder = DeepSDFDecoder(latent_dim=16, hidden_dim=128, num_layers=4).to(device)
    z = torch.zeros(1, 16, device=device)
    mesh = decode_mesh(decoder, z[0], resolution=32, device=device)
    print(f"decode smoke (untrained sphere init): mesh={'OK' if mesh and len(mesh.faces) else 'FAIL'}")
    if mesh is None or len(mesh.faces) == 0:
        print("FAIL: marching cubes could not extract a mesh — check scikit-image")
        sys.exit(1)

    print("PASS: preflight OK")


if __name__ == "__main__":
    main()
