"""Regenerate every report table and figure from a completed grid run.

Reads ``<output>/summary.json`` (produced by run_grid.py) and writes:
  * results.csv and results_table.tex   - the quantitative results,
  * degradation_generation.png          - Coverage / MMD / 1-NN vs N,
  * degradation_reconstruction.png      - Chamfer / IoU vs N,
  * loss_curves.png                     - Stage-1 and DDPM training losses,
  * gallery.png                         - reconstructions + Gaussian samples.

Usage:
    python scripts/make_figures.py --input outputs/smoke --figures figures
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.decoder import DeepSDFDecoder  # noqa: E402
from src.models.gaussian_prior import GaussianPrior  # noqa: E402
from src.sample import decode_point_cloud  # noqa: E402
from src.utils import load_checkpoint  # noqa: E402

GEN_METRICS = [("coverage", "Coverage (higher better)"),
               ("mmd", "MMD-CD (lower better)"),
               ("one_nn_acc", "1-NN acc (0.5 ideal)"),
               ("valid_ratio", "Valid ratio (higher better)")]


def load_summary(input_dir: Path) -> list[dict]:
    summary_path = input_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as handle:
            return json.load(handle)
    # Fall back to scanning individual cell files.
    records = []
    for metrics_file in sorted(input_dir.glob("N*_D*/metrics.json")):
        with open(metrics_file, encoding="utf-8") as handle:
            records.append(json.load(handle))
    return records


def flatten_records(summary: list[dict]) -> list[dict]:
    rows = []
    for cell in summary:
        recon = cell.get("reconstruction", {})
        for gen_name, gen in cell.get("generators", {}).items():
            rows.append(
                {
                    "N": cell["N"],
                    "D": cell["D"],
                    "generator": gen_name,
                    "recon_chamfer": recon.get("chamfer", float("nan")),
                    "recon_iou": recon.get("iou", float("nan")),
                    "coverage": gen.get("coverage", float("nan")),
                    "mmd": gen.get("mmd", float("nan")),
                    "one_nn_acc": gen.get("one_nn_acc", float("nan")),
                    "valid_ratio": gen.get("valid_ratio", float("nan")),
                }
            )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(rows: list[dict], path: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"$N$ & $D$ & Gen. & Recon-CD & IoU & Coverage & MMD & 1-NN & Valid \\",
        r"\midrule",
    ]
    for r in sorted(rows, key=lambda x: (x["N"], x["D"], x["generator"])):
        lines.append(
            f"{r['N']} & {r['D']} & {r['generator']} & "
            f"{r['recon_chamfer']:.4f} & {r['recon_iou']:.3f} & "
            f"{r['coverage']:.3f} & {r['mmd']:.4f} & {r['one_nn_acc']:.3f} & "
            f"{r['valid_ratio']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_generation_curves(rows: list[dict], path: Path) -> None:
    ds = sorted({r["D"] for r in rows})
    gens = sorted({r["generator"] for r in rows})
    fig, axes = plt.subplots(1, len(GEN_METRICS), figsize=(5 * len(GEN_METRICS), 4))
    if len(GEN_METRICS) == 1:
        axes = [axes]
    for ax, (key, title) in zip(axes, GEN_METRICS):
        for gen in gens:
            for d in ds:
                sub = sorted(
                    [r for r in rows if r["generator"] == gen and r["D"] == d],
                    key=lambda x: x["N"],
                )
                if not sub:
                    continue
                xs = [r["N"] for r in sub]
                ys = [r[key] for r in sub]
                ax.plot(xs, ys, marker="o", label=f"{gen}, D={d}")
        ax.set_xlabel("N (number of training shapes)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        # Mark the ideal 1-NN accuracy of 0.5 on its own panel.
        if key == "one_nn_acc" and any(r["one_nn_acc"] == r["one_nn_acc"] for r in rows):
            ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reconstruction_curves(rows: list[dict], path: Path) -> None:
    # Reconstruction is per (N, D); dedupe across generators.
    seen = {}
    for r in rows:
        seen[(r["N"], r["D"])] = (r["recon_chamfer"], r["recon_iou"])
    ds = sorted({d for (_, d) in seen})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for d in ds:
        items = sorted([(n, v) for (n, dd), v in seen.items() if dd == d])
        xs = [n for n, _ in items]
        axes[0].plot(xs, [v[0] for _, v in items], marker="o", label=f"D={d}")
        axes[1].plot(xs, [v[1] for _, v in items], marker="o", label=f"D={d}")
    axes[0].set_title("Reconstruction Chamfer (lower better)")
    axes[1].set_title("Reconstruction IoU (higher better)")
    for ax in axes:
        ax.set_xlabel("N (number of training shapes)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss_curves(summary: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for cell in summary:
        tag = f"N{cell['N']}_D{cell['D']}"
        hist = cell.get("stage1_history", [])
        if hist:
            axes[0].plot(
                [h["step"] for h in hist], [h["recon"] for h in hist], label=tag
            )
        dhist = cell.get("ddpm_history", [])
        if dhist:
            axes[1].plot(
                [h["step"] for h in dhist], [h["loss"] for h in dhist], label=tag
            )
    axes[0].set_title("Stage-1 reconstruction loss")
    axes[1].set_title("DDPM noise-prediction loss")
    for ax in axes:
        ax.set_xlabel("iteration")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _scatter(ax, pc: np.ndarray, title: str) -> None:
    if pc is not None:
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_axis_off()


def plot_gallery(summary: list[dict], input_dir: Path, path: Path) -> None:
    # Use the largest-N cell for the qualitative gallery.
    cell = max(summary, key=lambda c: c["N"])
    tag = f"N{cell['N']}_D{cell['D']}"
    ckpt_path = input_dir / tag / "checkpoint.pt"
    if not ckpt_path.exists():
        return
    ckpt = load_checkpoint(ckpt_path)

    latent_dim = ckpt["latent_dim"]
    cfg = ckpt["config"]
    decoder = DeepSDFDecoder(
        latent_dim=latent_dim,
        hidden_dim=int(cfg["decoder"]["hidden_dim"]),
        num_layers=int(cfg["decoder"]["num_layers"]),
        skip_in=tuple(cfg["decoder"]["skip_in"]),
        dropout_prob=float(cfg["decoder"]["dropout_prob"]),
        use_weight_norm=bool(cfg["decoder"]["use_weight_norm"]),
        use_tanh=bool(cfg["decoder"]["use_tanh"]),
        geometric_init=bool(cfg["decoder"].get("geometric_init", True)),
        init_radius=float(cfg["decoder"].get("init_radius", 0.5)),
    )
    decoder.load_state_dict(ckpt["decoder_state"])
    decoder.eval()

    codes = ckpt["codes_state"]["embedding.weight"]
    prior = GaussianPrior(
        covariance=str(cfg["gaussian"]["covariance"]), reg=float(cfg["gaussian"]["reg"])
    ).fit(codes)

    res = int(cfg["eval"]["recon_resolution"])
    n_pts = 1500
    gen = torch.Generator().manual_seed(0)

    columns = 4
    fig = plt.figure(figsize=(4 * columns, 8))
    # Row 1: reconstructions of training codes.
    for c in range(columns):
        ax = fig.add_subplot(2, columns, c + 1, projection="3d")
        idx = min(c, codes.shape[0] - 1)
        pc = decode_point_cloud(decoder, codes[idx], n_pts, resolution=res)
        _scatter(ax, pc, f"recon z_{idx}")
    # Row 2: Gaussian prior samples.
    z_samples = prior.sample(columns, generator=gen)
    for c in range(columns):
        ax = fig.add_subplot(2, columns, columns + c + 1, projection="3d")
        pc = decode_point_cloud(decoder, z_samples[c], n_pts, resolution=res)
        _scatter(ax, pc, f"gaussian sample {c}")
    fig.suptitle(f"Reconstructions & Gaussian samples ({tag})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="outputs/grid")
    parser.add_argument("--figures", type=str, default="figures")
    args = parser.parse_args()

    input_dir = Path(args.input)
    fig_dir = Path(args.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(input_dir)
    if not summary:
        print(f"[error] no results found in {input_dir}")
        return
    rows = flatten_records(summary)

    write_csv(rows, fig_dir / "results.csv")
    write_latex_table(rows, fig_dir / "results_table.tex")
    plot_generation_curves(rows, fig_dir / "degradation_generation.png")
    plot_reconstruction_curves(rows, fig_dir / "degradation_reconstruction.png")
    plot_loss_curves(summary, fig_dir / "loss_curves.png")
    plot_gallery(summary, input_dir, fig_dir / "gallery.png")
    print(f"Wrote tables and figures -> {fig_dir}")


if __name__ == "__main__":
    main()
