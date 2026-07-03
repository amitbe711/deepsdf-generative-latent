"""Optional data preparation / inspection utility.

  * ``--inspect-dir PATH``  : list loadable meshes in a directory (e.g. a
    ShapeNet chairs subset) and report how many pass the loader.
  * ``--cache PATH``        : preprocess a mesh directory into a cached ``.npz``
    of SDF samples for faster repeated runs.

Marching-cubes and analytic shapes need no preparation; this script is only for
real mesh datasets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.normalize import normalize_mesh_to_unit_sphere  # noqa: E402
from src.data.sdf_sampling import sample_sdf_from_mesh, sample_surface_points  # noqa: E402

_EXTS = (".obj", ".off", ".ply", ".stl", ".glb")


def inspect_dir(path: Path) -> None:
    files = sorted(p for p in path.rglob("*") if p.suffix.lower() in _EXTS)
    ok = 0
    for file in files:
        try:
            mesh = trimesh.load(file, force="mesh")
            if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces) > 0:
                ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {file.name}: {exc}")
    print(f"{ok}/{len(files)} meshes loadable in {path}")


def cache_dir(path: Path, out: Path, limit: int, num_points: int) -> None:
    files = sorted(p for p in path.rglob("*") if p.suffix.lower() in _EXTS)[:limit]
    rng = np.random.default_rng(0)
    points, sdfs, surfaces = [], [], []
    for file in files:
        mesh = normalize_mesh_to_unit_sphere(trimesh.load(file, force="mesh"))
        pts, sdf = sample_sdf_from_mesh(mesh, num_points=num_points, rng=rng)
        points.append(pts)
        sdfs.append(sdf)
        surfaces.append(sample_surface_points(mesh, 4096))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        points=np.stack(points),
        sdf=np.stack(sdfs),
        surface=np.stack(surfaces),
    )
    print(f"Cached {len(files)} shapes -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-dir", type=str, default=None)
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument("--out", type=str, default="data/cache.npz")
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--num-points", type=int, default=30000)
    args = parser.parse_args()

    if args.inspect_dir:
        inspect_dir(Path(args.inspect_dir))
    if args.cache:
        cache_dir(Path(args.cache), Path(args.out), args.limit, args.num_points)
    if not args.inspect_dir and not args.cache:
        parser.print_help()


if __name__ == "__main__":
    main()
