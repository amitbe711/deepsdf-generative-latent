"""Run the N x D experiment grid: Stage 1 + both Stage-2 generators per cell.

For every (N, D) cell this script:
  1. builds N training shapes and trains the DeepSDF auto-decoder,
  2. measures reconstruction quality (Chamfer + IoU) on the trained codes,
  3. fits the Gaussian prior and trains the latent DDPM,
  4. samples each generator, decodes to point clouds, and measures generation
     quality (Coverage / MMD / 1-NN) against a held-out reference set,
  5. writes metrics JSON + loss histories + checkpoints to the output dir.

Usage:
    python scripts/run_grid.py --config configs/base.yaml --output outputs/grid
    python scripts/run_grid.py --config configs/smoke.yaml --output outputs/smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import ShapeSDFDataset, _load_meshes_from_dir, build_shape_collection  # noqa: E402
from src.metrics.generation import (  # noqa: E402
    chamfer_matrix,
    coverage,
    minimum_matching_distance,
    one_nn_accuracy,
)
from src.metrics.reconstruction import chamfer_distance, iou_from_meshes  # noqa: E402
from src.sample import decode_mesh, decode_point_cloud  # noqa: E402
from src.train import fit_gaussian, fit_gmm, train_ddpm, train_stage1  # noqa: E402
from src.utils import load_config, save_checkpoint, seed_everything  # noqa: E402
from src.utils.log import Phase, format_duration, status  # noqa: E402
from src.data.sdf_sampling import sample_surface_points  # noqa: E402


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return requested


def build_reference_clouds(
    cfg, num_reference: int, surface_points: int, train_count: int = 0
) -> list[np.ndarray]:
    """Held-out shapes used as the reference distribution for generation metrics."""
    if cfg.data.source == "mesh_dir":
        mesh_dir = Path(cfg.data.mesh_dir)
        meshes = _load_meshes_from_dir(mesh_dir, limit=None)
        ref_meshes = meshes[train_count : train_count + num_reference]
        if len(ref_meshes) >= num_reference:
            return [
                sample_surface_points(mesh, surface_points) for mesh in ref_meshes
            ]
        print(
            f"[warn] only {len(meshes) - train_count} held-out meshes in {mesh_dir}; "
            "using synthetic chairs as the reference set."
        )

    ref_cfg = cfg.merge({"seed": int(cfg.seed) + 10_000, "data.source": "synthetic"})
    collection = build_shape_collection(ref_cfg, num_reference)
    clouds = []
    for shape in collection:
        mesh = shape["mesh"]
        clouds.append(sample_surface_points(mesh, surface_points))
    return clouds


def evaluate_reconstruction(
    cfg, decoder, codes, dataset, device, *, prefix: str | None = None, verbose: bool = False
) -> dict[str, float]:
    max_shapes = min(int(cfg.eval.max_recon_shapes), dataset.num_shapes)
    idxs = list(range(max_shapes))
    surface = dataset.surface_clouds()
    meshes = dataset.meshes()
    res = int(cfg.eval.recon_resolution)
    iou_res = int(cfg.eval.iou_resolution)
    n_pts = int(cfg.eval.surface_points)

    if verbose:
        status(f"reconstruction eval on {max_shapes} shapes", prefix=prefix)

    cds, ious = [], []
    for i in idxs:
        if verbose and (i == 0 or (i + 1) % max(1, max_shapes // 5) == 0 or i == max_shapes - 1):
            status(f"reconstruction {i + 1}/{max_shapes}", prefix=prefix)
        z = codes.embedding.weight[i].detach()
        pred_mesh = decode_mesh(decoder, z, resolution=res, device=device)
        if pred_mesh is None or len(pred_mesh.faces) == 0:
            continue
        pred_pc = sample_surface_points(pred_mesh, n_pts)
        cds.append(chamfer_distance(pred_pc, surface[i]))
        if meshes[i] is not None:
            ious.append(iou_from_meshes(pred_mesh, meshes[i], resolution=iou_res))
    return {
        "chamfer": float(np.mean(cds)) if cds else float("nan"),
        "iou": float(np.mean(ious)) if ious else float("nan"),
        "num_evaluated": len(cds),
    }


def evaluate_generator(
    cfg,
    decoder,
    sampler_fn,
    reference_clouds,
    device,
    *,
    prefix: str | None = None,
    verbose: bool = False,
    generator_name: str = "generator",
) -> dict[str, float]:
    num_gen = int(cfg.eval.num_generated)
    res = int(cfg.eval.recon_resolution)
    n_pts = int(cfg.eval.surface_points)

    if verbose:
        status(f"{generator_name}: sampling {num_gen} latents", prefix=prefix)

    z_samples = sampler_fn(num_gen)
    gen_clouds = []
    for j in range(z_samples.shape[0]):
        if verbose and (j == 0 or (j + 1) % max(1, num_gen // 5) == 0 or j == num_gen - 1):
            status(f"{generator_name}: decode {j + 1}/{num_gen}", prefix=prefix)
        pc = decode_point_cloud(
            decoder, z_samples[j], num_points=n_pts, resolution=res, device=device
        )
        if pc is not None:
            gen_clouds.append(pc)

    valid_ratio = len(gen_clouds) / max(num_gen, 1)
    if len(gen_clouds) == 0:
        return {
            "coverage": float("nan"),
            "mmd": float("nan"),
            "one_nn_acc": float("nan"),
            "valid_ratio": valid_ratio,
        }

    mat = chamfer_matrix(gen_clouds, reference_clouds)
    return {
        "coverage": coverage(mat),
        "mmd": minimum_matching_distance(mat),
        "one_nn_acc": one_nn_accuracy(gen_clouds, reference_clouds),
        "valid_ratio": valid_ratio,
    }


def run_cell(
    cfg,
    num_shapes: int,
    latent_dim: int,
    output_dir: Path,
    device: str,
    *,
    cell_index: int = 1,
    cell_total: int = 1,
    verbose: bool = True,
) -> dict:
    seed_everything(int(cfg.seed))
    tag = f"N{num_shapes}_D{latent_dim}"
    status(
        f"grid cell {cell_index}/{cell_total}: {tag} on {device} "
        f"(generators={list(cfg.grid.generators)})",
    )
    t0 = time.time()

    with Phase("build training shapes", prefix=tag):
        collection = build_shape_collection(cfg, num_shapes)
        dataset = ShapeSDFDataset(collection)
        status(
            f"loaded {dataset.num_shapes} shapes (source={cfg.data.source})",
            prefix=tag,
        )

    with Phase("stage-1 auto-decoder", prefix=tag):
        stage1 = train_stage1(
            cfg,
            dataset,
            latent_dim,
            device=device,
            progress=False,
            verbose=verbose,
            prefix=tag,
        )
    decoder, codes = stage1["decoder"], stage1["codes"]

    with Phase("reconstruction metrics", prefix=tag):
        recon = evaluate_reconstruction(
            cfg, decoder, codes, dataset, device, prefix=tag, verbose=verbose
        )
    status(f"reconstruction: chamfer={recon['chamfer']:.4f} iou={recon['iou']:.3f}", prefix=tag)

    with Phase("reference set", prefix=tag):
        reference = build_reference_clouds(
            cfg,
            int(cfg.eval.num_reference),
            int(cfg.eval.surface_points),
            train_count=num_shapes,
        )
        status(f"reference clouds: {len(reference)}", prefix=tag)

    results: dict = {
        "N": num_shapes,
        "D": latent_dim,
        "reconstruction": recon,
        "stage1_history": stage1["history"],
        "generators": {},
    }

    code_tensor = codes.embedding.weight.detach().cpu()

    if "gaussian" in cfg.grid.generators:
        with Phase("gaussian prior", prefix=tag):
            prior = fit_gaussian(cfg, code_tensor)
        with Phase("gaussian generation eval", prefix=tag):
            gen = torch.Generator().manual_seed(int(cfg.seed) + 7)
            metrics = evaluate_generator(
                cfg,
                decoder,
                lambda n: prior.sample(n, generator=gen),
                reference,
                device,
                prefix=tag,
                verbose=verbose,
                generator_name="gaussian",
            )
        results["generators"]["gaussian"] = metrics
        status(
            f"gaussian: coverage={metrics['coverage']:.3f} mmd={metrics['mmd']:.4f} "
            f"1-nn={metrics['one_nn_acc']:.3f} valid={metrics['valid_ratio']:.2f}",
            prefix=tag,
        )

    if "gmm" in cfg.grid.generators:
        with Phase("gmm prior (EM)", prefix=tag):
            gmm = fit_gmm(cfg, code_tensor)
        with Phase("gmm generation eval", prefix=tag):
            gen = torch.Generator().manual_seed(int(cfg.seed) + 8)
            metrics = evaluate_generator(
                cfg,
                decoder,
                lambda n: gmm.sample(n, generator=gen),
                reference,
                device,
                prefix=tag,
                verbose=verbose,
                generator_name="gmm",
            )
        results["generators"]["gmm"] = metrics
        status(
            f"gmm: coverage={metrics['coverage']:.3f} mmd={metrics['mmd']:.4f} "
            f"1-nn={metrics['one_nn_acc']:.3f} valid={metrics['valid_ratio']:.2f}",
            prefix=tag,
        )

    if "ddpm" in cfg.grid.generators:
        with Phase("ddpm training", prefix=tag):
            ddpm_out = train_ddpm(
                cfg,
                code_tensor,
                device=device,
                progress=False,
                verbose=verbose,
                prefix=tag,
            )
        ddpm = ddpm_out["model"]
        with Phase("ddpm generation eval", prefix=tag):
            metrics = evaluate_generator(
                cfg,
                decoder,
                lambda n: ddpm.sample(n, device=device).cpu(),
                reference,
                device,
                prefix=tag,
                verbose=verbose,
                generator_name="ddpm",
            )
        results["generators"]["ddpm"] = metrics
        results["ddpm_history"] = ddpm_out["history"]
        status(
            f"ddpm: coverage={metrics['coverage']:.3f} mmd={metrics['mmd']:.4f} "
            f"1-nn={metrics['one_nn_acc']:.3f} valid={metrics['valid_ratio']:.2f}",
            prefix=tag,
        )

    results["seconds"] = time.time() - t0

    with Phase("checkpoint + metrics", prefix=tag):
        cell_dir = output_dir / tag
        save_checkpoint(
            cell_dir / "checkpoint.pt",
            {
                "config": dict(cfg),
                "decoder_state": decoder.state_dict(),
                "codes_state": codes.state_dict(),
                "latent_dim": latent_dim,
                "num_shapes": num_shapes,
            },
        )
        with open(cell_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

    status(f"cell finished in {format_duration(results['seconds'])}", prefix=tag)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--output", type=str, default="outputs/grid")
    parser.add_argument("--only-N", type=int, default=None, help="run a single N")
    parser.add_argument("--only-D", type=int, default=None, help="run a single D")
    parser.add_argument("--quiet", action="store_true", help="suppress step-by-step status")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(str(cfg.device))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    grid_n = [args.only_N] if args.only_N else list(cfg.grid.N)
    grid_d = [args.only_D] if args.only_D else list(cfg.grid.D)
    cells = [(n, d) for n in grid_n for d in grid_d]
    total = len(cells)

    status(
        f"starting grid: config={args.config} device={device} "
        f"cells={total} N={grid_n} D={grid_d} output={output_dir}"
    )

    summary = []
    grid_t0 = time.time()
    for idx, (num_shapes, latent_dim) in enumerate(cells, start=1):
        results = run_cell(
            cfg,
            num_shapes,
            latent_dim,
            output_dir,
            device,
            cell_index=idx,
            cell_total=total,
            verbose=verbose,
        )
        summary.append(results)
        elapsed = time.time() - grid_t0
        avg = elapsed / idx
        remaining = avg * (total - idx)
        status(
            f"grid progress {idx}/{total} "
            f"elapsed={format_duration(elapsed)} eta={format_duration(remaining)}"
        )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    status(f"grid complete in {format_duration(time.time() - grid_t0)} -> {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
