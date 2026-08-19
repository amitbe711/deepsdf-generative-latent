"""Re-measure a finished grid at different eval settings, without retraining.

Stage-1 training is the expensive part of run_grid.py and it does not depend on
the evaluation protocol, so raising the sample counts does not require redoing
it. This script reloads each cell's checkpoint and recomputes only the metrics.

The motivating problem: the default protocol scores 40 generated shapes against
40 references. Coverage is then quantised to 1/40 = 0.025, which is as large as
the gaps being reported between priors, and 1-NN accuracy is biased toward 1.0
at that sample size because in a sparse set a shape's nearest neighbour is
usually from its own half. Both make the headline "1-NN ~ 0.95" partly an
artefact of the measurement rather than a property of the model.

What is *not* reproduced exactly: the DDPM is not checkpointed, so it is
retrained here from the saved latent codes. Its numbers will move slightly even
at identical eval settings; the Gaussian and GMM priors are refit in closed form
from the same codes and are deterministic.

The previous metrics.json / summary.json are preserved as *_pre_reeval.json the
first time this runs, so the original numbers stay recoverable.

Usage:
    python scripts/reeval_grid.py --output outputs/shapenet_scoped \
        --num-generated 200 --num-reference 200 --iou-resolution 64 \
        --max-recon-shapes 25
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_grid import (  # noqa: E402
    _decode_device,
    build_reference_clouds,
    evaluate_generator,
    evaluate_reconstruction,
    resolve_device,
)

from src.data.dataset import ShapeSDFDataset, _mesh_files, build_shape_collection  # noqa: E402
from src.models.decoder import DeepSDFDecoder  # noqa: E402
from src.models.latent_codes import LatentCodes  # noqa: E402
from src.train import fit_gaussian, fit_gmm, train_ddpm  # noqa: E402
from src.utils import load_checkpoint, seed_everything  # noqa: E402
from src.utils.config import AttrDict, load_config  # noqa: E402
from src.utils.log import Phase, format_duration, status  # noqa: E402


def _fill_missing(base: dict, extra: dict) -> dict:
    """Recursively add keys from ``extra`` that ``base`` does not already define.

    Only ever *adds*. Anything the checkpoint recorded wins, so supplying a config
    can never silently re-describe the model that was actually trained.
    """
    out = dict(base)
    for key, value in extra.items():
        if key not in out:
            out[key] = value
        elif isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _fill_missing(out[key], value)
    return out


def cell_config(ckpt: dict, overrides: dict, fallback: dict | None) -> AttrDict:
    raw = ckpt["config"]
    if fallback:
        raw = _fill_missing(dict(raw), fallback)
    return AttrDict(raw).merge(overrides)


def check_generators(cfg: AttrDict, tag: str) -> list[str]:
    """Requested generators that have no hyperparameter block in this config."""
    return [g for g in list(cfg.grid.generators) if g not in cfg]


def build_decoder(ckpt: dict) -> DeepSDFDecoder:
    dec_cfg = ckpt["config"]["decoder"]
    decoder = DeepSDFDecoder(
        latent_dim=ckpt["latent_dim"],
        hidden_dim=int(dec_cfg["hidden_dim"]),
        num_layers=int(dec_cfg["num_layers"]),
        skip_in=tuple(dec_cfg["skip_in"]),
        dropout_prob=float(dec_cfg["dropout_prob"]),
        use_weight_norm=bool(dec_cfg["use_weight_norm"]),
        use_tanh=bool(dec_cfg["use_tanh"]),
        geometric_init=bool(dec_cfg.get("geometric_init", True)),
        init_radius=float(dec_cfg.get("init_radius", 0.5)),
    )
    decoder.load_state_dict(ckpt["decoder_state"])
    decoder.eval()
    return decoder


def reeval_cell(
    ckpt_path: Path,
    overrides: dict,
    device: str,
    *,
    previous: dict | None = None,
    skip_recon: bool = False,
    verbose: bool = True,
    fallback_config: dict | None = None,
    reference_cache: dict | None = None,
) -> dict:
    ckpt = load_checkpoint(ckpt_path)
    cfg = cell_config(ckpt, overrides, fallback_config)
    num_shapes = int(ckpt["num_shapes"])
    latent_dim = int(ckpt["latent_dim"])
    tag = f"N{num_shapes}_D{latent_dim}"

    # Same seed as run_cell so the shapes, their SDF samples and the prior draws
    # line up with the training run this checkpoint came from.
    seed_everything(int(cfg.seed))

    decoder = build_decoder(ckpt)
    codes = LatentCodes(
        num_shapes=num_shapes,
        latent_dim=latent_dim,
        init_std=float(cfg.stage1.code_init_std),
    )
    codes.load_state_dict(ckpt["codes_state"])
    code_tensor = codes.embedding.weight.detach().cpu()

    decode_device = _decode_device(cfg, device)
    decoder = decoder.to(decode_device)

    results: dict = {"N": num_shapes, "D": latent_dim, "generators": {}}
    # Training curves belong to the run that produced the checkpoint; carry them
    # across so the loss figures still have data after the summary is rewritten.
    for key in ("stage1_history", "ddpm_history"):
        if previous and key in previous:
            results[key] = previous[key]

    if skip_recon:
        results["reconstruction"] = (previous or {}).get("reconstruction", {})
        status("reusing previous reconstruction metrics", prefix=tag)
    else:
        # evaluate_reconstruction only ever looks at the first max_recon_shapes
        # shapes (matched to the same-indexed latent codes), so that is all that
        # needs rebuilding -- not the full N the checkpoint was trained on. For
        # N=150 with max_recon_shapes=25 that is a 6x cut in the slowest step
        # (SDF-sampling meshes from disk).
        #
        # Which *mesh file* becomes shape i is deterministic (sorted directory
        # order, offset=0), so this is always the correct chair regardless of
        # how many shapes are requested -- verified directly, not assumed.
        # What is NOT reproducible, with or without this change, is the exact
        # surface point cloud sampled from that mesh: sample_surface_points
        # goes through trimesh's own OS-entropy-seeded generator, which
        # seed_everything() cannot reach, so every rebuild (even at the
        # original N) redraws different points from the same true surface.
        # That's harmless Monte Carlo noise in the Chamfer distance -- the
        # IoU ground truth is the repaired mesh itself (voxelize + marching
        # cubes, no randomness involved) and is identical either way.
        recon_shapes = min(num_shapes, int(cfg.eval.max_recon_shapes))
        with Phase("rebuild shapes for reconstruction eval", prefix=tag):
            dataset = ShapeSDFDataset(
                build_shape_collection(cfg, recon_shapes, verbose=verbose, prefix=tag)
            )
        with Phase("reconstruction metrics", prefix=tag):
            results["reconstruction"] = evaluate_reconstruction(
                cfg,
                decoder,
                codes,
                dataset,
                device,
                decode_device=decode_device,
                prefix=tag,
                verbose=verbose,
            )
        recon = results["reconstruction"]
        status(
            f"reconstruction: chamfer={recon['chamfer']:.4f} iou={recon['iou']:.3f} "
            f"over {recon['num_evaluated']} shapes",
            prefix=tag,
        )
        del dataset

    # Cells sharing an N share a reference set (meshes [N : N+num_reference)), so
    # cache it: rebuilding means re-reading hundreds of meshes from Drive.
    ref_key = (num_shapes, int(cfg.eval.num_reference), int(cfg.eval.surface_points))
    if reference_cache is not None and ref_key in reference_cache:
        reference = reference_cache[ref_key]
        status(f"reference clouds: {len(reference)} (cached)", prefix=tag)
    else:
        with Phase("reference set", prefix=tag):
            reference = build_reference_clouds(
                cfg,
                int(cfg.eval.num_reference),
                int(cfg.eval.surface_points),
                train_count=num_shapes,
            )
            status(f"reference clouds: {len(reference)}", prefix=tag)
        if reference_cache is not None:
            reference_cache[ref_key] = reference

    # From the checkpoint's own config unless overridden: a checkpoint written
    # before a prior existed will not list it, and silently scoring two of three
    # priors is easy to miss in a long log.
    generators = list(cfg.grid.generators)
    status(f"generators: {generators}", prefix=tag)

    if "gaussian" in generators:
        prior = fit_gaussian(cfg, code_tensor)
        gen = torch.Generator().manual_seed(int(cfg.seed) + 7)
        with Phase("gaussian generation eval", prefix=tag):
            results["generators"]["gaussian"] = evaluate_generator(
                cfg, decoder, lambda n: prior.sample(n, generator=gen), reference,
                device, decode_device=decode_device, prefix=tag, verbose=verbose,
                generator_name="gaussian",
            )

    if "gmm" in generators:
        gmm = fit_gmm(cfg, code_tensor)
        gen = torch.Generator().manual_seed(int(cfg.seed) + 8)
        with Phase("gmm generation eval", prefix=tag):
            results["generators"]["gmm"] = evaluate_generator(
                cfg, decoder, lambda n: gmm.sample(n, generator=gen), reference,
                device, decode_device=decode_device, prefix=tag, verbose=verbose,
                generator_name="gmm",
            )

    if "ddpm" in generators:
        with Phase("ddpm retraining (not checkpointed)", prefix=tag):
            ddpm_out = train_ddpm(
                cfg, code_tensor, device=device, progress=False,
                verbose=verbose, prefix=tag,
            )
        results["ddpm_history"] = ddpm_out["history"]
        ddpm = ddpm_out["model"]
        with Phase("ddpm generation eval", prefix=tag):
            results["generators"]["ddpm"] = evaluate_generator(
                cfg, decoder, lambda n: ddpm.sample(n, device=device).cpu(), reference,
                device, decode_device=decode_device, prefix=tag, verbose=verbose,
                generator_name="ddpm",
            )

    for name, m in results["generators"].items():
        status(
            f"{name}: coverage={m['coverage']:.3f} mmd={m['mmd']:.4f} "
            f"1-nn={m['one_nn_acc']:.3f} valid={m['valid_ratio']:.2f}",
            prefix=tag,
        )
    return results


def _backup_once(path: Path) -> None:
    """Preserve the pre-reeval file, but never overwrite an existing backup."""
    if path.exists():
        backup = path.with_name(f"{path.stem}_pre_reeval{path.suffix}")
        if not backup.exists():
            backup.write_bytes(path.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True, help="finished grid dir")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML supplying hyperparameter blocks the checkpoints predate (e.g. a "
        "prior added after they were written). Never overrides recorded values",
    )
    parser.add_argument("--num-generated", type=int, default=200)
    parser.add_argument("--num-reference", type=int, default=200)
    parser.add_argument("--iou-resolution", type=int, default=64)
    parser.add_argument("--max-recon-shapes", type=int, default=25)
    parser.add_argument("--recon-resolution", type=int, default=None)
    parser.add_argument("--decode-device", type=str, default=None)
    parser.add_argument("--only-N", type=int, default=None)
    parser.add_argument("--only-D", type=int, default=None)
    parser.add_argument(
        "--generators",
        type=str,
        default=None,
        help="comma-separated override, e.g. gaussian,gmm,ddpm. Defaults to "
        "whatever the checkpoint's config recorded",
    )
    parser.add_argument(
        "--skip-recon",
        action="store_true",
        help="only redo generation metrics; keeps the previous reconstruction "
        "numbers and avoids re-sampling the training meshes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-score cells that already carry these exact eval settings",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    if not output_dir.exists():
        raise SystemExit(f"{output_dir} does not exist")
    verbose = not args.quiet

    protocol = {
        "num_generated": args.num_generated,
        "num_reference": args.num_reference,
        "iou_resolution": args.iou_resolution,
        "max_recon_shapes": args.max_recon_shapes,
    }
    overrides = {f"eval.{k}": v for k, v in protocol.items()}
    if args.recon_resolution:
        overrides["eval.recon_resolution"] = args.recon_resolution
    if args.decode_device:
        overrides["eval.decode_device"] = args.decode_device
    if args.generators:
        overrides["grid.generators"] = [g.strip() for g in args.generators.split(",")]

    cells = []
    for ckpt_path in sorted(output_dir.glob("N*_D*/checkpoint.pt")):
        tag = ckpt_path.parent.name
        n = int(tag.split("_")[0][1:])
        d = int(tag.split("_")[1][1:])
        if args.only_N and n != args.only_N:
            continue
        if args.only_D and d != args.only_D:
            continue
        cells.append((n, d, ckpt_path))
    if not cells:
        raise SystemExit(f"no checkpoints matched under {output_dir}")

    fallback = dict(load_config(args.config)) if args.config else None

    device = resolve_device("cuda")
    status(
        f"re-evaluating {len(cells)} cells in {output_dir} on {device}: "
        f"{args.num_generated} generated vs {args.num_reference} reference, "
        f"IoU at {args.iou_resolution}^3 over {args.max_recon_shapes} shapes"
    )

    # Preflight every cell before doing any work: a checkpoint that predates a
    # prior has no hyperparameters for it, and discovering that after the first
    # cell has already been scored wastes hours.
    problems = []
    for n, d, ckpt_path in cells:
        cfg = cell_config(load_checkpoint(ckpt_path), overrides, fallback)
        missing = check_generators(cfg, f"N{n}_D{d}")
        if missing:
            problems.append(f"  N{n}_D{d}: no config block for {missing}")
        # The reference set is meshes [N : N + num_reference). If the directory
        # cannot supply them, build_reference_clouds silently substitutes
        # *synthetic* chairs -- the metrics would still be produced, but against
        # a different distribution, making them incomparable to the real-data
        # numbers. Refuse rather than let that reach a results table.
        if str(cfg.data.source) == "mesh_dir":
            available = len(_mesh_files(Path(cfg.data.mesh_dir)))
            need = n + int(cfg.eval.num_reference)
            if available < need:
                problems.append(
                    f"  N{n}_D{d}: needs {need} meshes ({n} train + "
                    f"{int(cfg.eval.num_reference)} reference) but "
                    f"{cfg.data.mesh_dir} has {available}"
                )
    if problems:
        raise SystemExit(
            "preflight failed:\n"
            + "\n".join(problems)
            + "\nSupply missing hyperparameters with --config <yaml>, and/or lower "
            "--num-reference so the held-out set fits the available meshes."
        )

    summary_by_tag: dict[str, dict] = {}
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        for entry in json.loads(summary_path.read_text(encoding="utf-8")):
            summary_by_tag[f"N{entry['N']}_D{entry['D']}"] = entry

    t0 = time.time()
    reference_cache: dict = {}
    for idx, (n, d, ckpt_path) in enumerate(cells, start=1):
        tag = f"N{n}_D{d}"
        status(f"cell {idx}/{len(cells)}: {tag}")
        metrics_path = ckpt_path.parent / "metrics.json"
        previous = None
        if metrics_path.exists():
            previous = json.loads(metrics_path.read_text(encoding="utf-8"))

        # Resumable across sessions, like run_grid.py: a Colab runtime that dies
        # midway can just re-run this command unchanged.
        if not args.force and previous and previous.get("eval_protocol") == protocol:
            status(f"{tag} already scored at these settings -- skipping (--force to redo)")
            summary_by_tag[tag] = previous
            continue

        results = reeval_cell(
            ckpt_path,
            overrides,
            device,
            previous=previous,
            skip_recon=args.skip_recon,
            verbose=verbose,
            fallback_config=fallback,
            reference_cache=reference_cache,
        )
        results["eval_protocol"] = protocol

        _backup_once(metrics_path)
        metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        summary_by_tag[tag] = results
        _backup_once(summary_path)
        ordered = sorted(summary_by_tag.values(), key=lambda r: (r["N"], r["D"]))
        summary_path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")

        elapsed = time.time() - t0
        status(
            f"progress {idx}/{len(cells)} elapsed={format_duration(elapsed)} "
            f"eta={format_duration(elapsed / idx * (len(cells) - idx))}"
        )

    status(f"re-evaluation complete in {format_duration(time.time() - t0)} -> {summary_path}")


if __name__ == "__main__":
    main()
