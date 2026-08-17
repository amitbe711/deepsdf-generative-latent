"""Regenerate every report table and figure from a completed grid run.

Reads ``<output>/summary.json`` (produced by run_grid.py) and writes:
  * results.csv and results_table.tex   - the quantitative results,
  * degradation_generation.png          - Coverage / MMD / 1-NN vs N,
  * degradation_reconstruction.png      - Chamfer / IoU vs N,
  * loss_curves.png                     - Stage-1 and DDPM training losses,
  * gallery.png                         - GT / recon / one row per generator.

The experiment grid is deliberately ragged (one latent dimension carries the
N-sweep; any other D appears at a single N as an ablation), so the degradation
figures plot only the sweep dimension and the table reports the ablation in a
separate block. See ``main_sweep_dim``.

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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import _load_meshes_from_dir  # noqa: E402
from src.data.normalize import make_watertight, normalize_mesh_to_unit_sphere  # noqa: E402
from src.models.decoder import DeepSDFDecoder  # noqa: E402
from src.sample import decode_mesh  # noqa: E402
from src.train import fit_gaussian, fit_gmm, train_ddpm  # noqa: E402
from src.utils import load_checkpoint  # noqa: E402
from src.utils.config import AttrDict  # noqa: E402

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


def main_sweep_dim(rows: list[dict]) -> int:
    """The D that carries the N-sweep (the one measured at the most values of N).

    The experiment is deliberately ragged: one D spans every N, while any other D
    appears at a single N as an ablation. Plotting them together would draw those
    ablation cells as isolated one-point "curves", so the degradation figures use
    only this D and the ablation is reported in the table instead.
    """
    per_dim = {}
    for r in rows:
        per_dim.setdefault(r["D"], set()).add(r["N"])
    if not per_dim:
        return 0
    # Most N values wins; tie-break on the smaller D (the validated default).
    return min(per_dim, key=lambda d: (-len(per_dim[d]), d))


def _table_row(r: dict) -> str:
    return (
        f"{r['N']} & {r['D']} & {r['generator']} & "
        f"{r['recon_chamfer']:.4f} & {r['recon_iou']:.3f} & "
        f"{r['coverage']:.3f} & {r['mmd']:.4f} & {r['one_nn_acc']:.3f} & "
        f"{r['valid_ratio']:.2f} \\\\"
    )


def write_latex_table(rows: list[dict], path: Path) -> None:
    main_d = main_sweep_dim(rows)
    ordered = sorted(rows, key=lambda x: (x["N"], x["D"], x["generator"]))
    sweep = [r for r in ordered if r["D"] == main_d]
    ablation = [r for r in ordered if r["D"] != main_d]

    lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"$N$ & $D$ & Gen. & Recon-CD & IoU & Coverage & MMD & 1-NN & Valid \\",
        r"\midrule",
        rf"\multicolumn{{9}}{{l}}{{\textit{{Main sweep ($D={main_d}$)}}}} \\",
    ]
    lines += [_table_row(r) for r in sweep]
    if ablation:
        lines += [
            r"\midrule",
            r"\multicolumn{9}{l}{\textit{Latent-dimension ablation}} \\",
        ]
        lines += [_table_row(r) for r in ablation]
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_generation_curves(rows: list[dict], path: Path) -> None:
    # A wide 1x4 strip: spanning both columns of a two-column paper, this costs a
    # quarter of the vertical space a 2x2 block would. Fonts are enlarged to stay
    # legible after the ~2x reduction to \textwidth.
    main_d = main_sweep_dim(rows)
    sweep = [r for r in rows if r["D"] == main_d]
    gens = sorted({r["generator"] for r in sweep})
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, (key, title) in zip(axes.flat, GEN_METRICS):
        for gen in gens:
            sub = sorted(
                [r for r in sweep if r["generator"] == gen], key=lambda x: x["N"]
            )
            if not sub:
                continue
            ax.plot(
                [r["N"] for r in sub],
                [r[key] for r in sub],
                marker="o",
                label=gen,
            )
        ax.set_xlabel("N", fontsize=14)
        ax.set_title(title, fontsize=15)
        ax.tick_params(labelsize=12)
        ax.grid(True, alpha=0.3)
        # Mark the ideal 1-NN accuracy of 0.5 on its own panel.
        if key == "one_nn_acc" and any(r["one_nn_acc"] == r["one_nn_acc"] for r in sweep):
            ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reconstruction_curves(rows: list[dict], path: Path) -> None:
    # Reconstruction is a Stage-1 property, shared by the generators in a cell.
    main_d = main_sweep_dim(rows)
    seen = {}
    for r in rows:
        if r["D"] == main_d:
            seen[r["N"]] = (r["recon_chamfer"], r["recon_iou"])
    items = sorted(seen.items())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    xs = [n for n, _ in items]
    axes[0].plot(xs, [v[0] for _, v in items], marker="o")
    axes[1].plot(xs, [v[1] for _, v in items], marker="o")
    axes[0].set_title("Reconstruction Chamfer (lower better)")
    axes[1].set_title("Reconstruction IoU (higher better)")
    for ax in axes:
        ax.set_xlabel("N (number of training shapes)")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Reconstruction vs. N (D={main_d})")
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


# Shaded-solid rendering constants. A sparse scatter of surface points reads as
# a cloud of specks at figure scale; shading the actual triangles makes it legible
# as a chair, which is the whole point of a qualitative figure.
LIGHT_DIR = np.array([0.4, -0.8, 0.9]) / np.linalg.norm([0.4, -0.8, 0.9])
FACE_COLOR = np.array([0.55, 0.70, 0.90])
UP_AXIS_SWAP = [0, 2, 1]  # ShapeNet is Y-up; matplotlib's 3rd axis is vertical


def _render_mesh(ax, mesh, title: str | None = None) -> None:
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    # 3D axes reserve generous margins around the data box; with the axes hidden
    # that reads as a tiny shape floating in whitespace, so zoom in on the box.
    # Chairs are wider than they are tall, so the panel is shorter than square and
    # the zoom is set to fill it without clipping a shape that reaches the corners.
    ax.set_box_aspect((1, 1, 1), zoom=1.75)
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-60)
    if title is not None:
        ax.set_title(title, fontsize=22, pad=4)
    if mesh is None or len(mesh.faces) == 0:
        ax.text2D(
            0.5, 0.5, "fail", transform=ax.transAxes, fontsize=20,
            ha="center", va="center", color="0.4",
        )
        return
    verts = mesh.vertices[:, UP_AXIS_SWAP]
    normals = mesh.face_normals[:, UP_AXIS_SWAP]
    shade = np.clip(normals @ LIGHT_DIR, 0.25, 1.0)
    colors = np.clip(shade[:, None] * FACE_COLOR[None, :], 0, 1)
    ax.add_collection3d(
        Poly3DCollection(verts[mesh.faces], facecolor=colors, edgecolor="none")
    )


def _ground_truth_meshes(cfg: dict, count: int) -> list:
    """Training meshes under the same normalization + repair used for supervision."""
    if str(cfg.get("data", {}).get("source", "")) != "mesh_dir":
        return []
    mesh_dir = Path(str(cfg["data"]["mesh_dir"]))
    if not mesh_dir.exists():
        return []
    pitch = float(cfg["data"].get("watertight_pitch", 1.0 / 64.0))
    meshes = []
    for mesh in _load_meshes_from_dir(mesh_dir, limit=count):
        mesh = normalize_mesh_to_unit_sphere(mesh)
        if not mesh.is_watertight:
            mesh = normalize_mesh_to_unit_sphere(make_watertight(mesh, pitch=pitch))
        meshes.append(mesh)
    return meshes


def plot_gallery(
    summary: list[dict],
    input_dir: Path,
    path: Path,
    cell_tag: str | None = None,
    examples: int = 3,
) -> None:
    """Render the qualitative gallery for one cell.

    Laid out with the five categories (ground truth, reconstruction, and one per
    prior) as *columns* and ``examples`` independent draws as rows. Categories
    read once as column headers rather than being repeated in every panel title,
    and the result is wider than it is tall, which is what a two-column paper can
    actually afford to print at a legible size.

    ``cell_tag`` selects the cell as ``N<n>_D<d>``; defaults to the largest-N
    cell, which is the best-trained one.
    """
    if cell_tag is not None:
        by_tag = {f"N{c['N']}_D{c['D']}": c for c in summary}
        if cell_tag not in by_tag:
            raise SystemExit(
                f"gallery cell {cell_tag!r} not in summary; have {sorted(by_tag)}"
            )
        cell = by_tag[cell_tag]
    else:
        cell = max(summary, key=lambda c: c["N"])
    tag = f"N{cell['N']}_D{cell['D']}"
    ckpt_path = input_dir / tag / "checkpoint.pt"
    if not ckpt_path.exists():
        return
    ckpt = load_checkpoint(ckpt_path)

    cfg = ckpt["config"]
    decoder = DeepSDFDecoder(
        latent_dim=ckpt["latent_dim"],
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = decoder.to(device)
    decoder.eval()

    codes = ckpt["codes_state"]["embedding.weight"]
    attr_cfg = AttrDict(cfg)
    # Coarse grids look blocky once rendered as a solid rather than as dots.
    res = max(int(cfg["eval"]["recon_resolution"]), 64)

    def decode(z):
        return decode_mesh(decoder, z.detach().cpu(), resolution=res, device=device)

    # One entry per column: (header, meshes down the rows).
    cols: list[tuple[str, list]] = []
    gt = _ground_truth_meshes(cfg, examples)
    if gt:
        cols.append(("Ground truth", gt))
    cols.append(
        ("Reconstruction", [decode(codes[min(i, codes.shape[0] - 1)]) for i in range(examples)])
    )

    generators = list(cfg.get("grid", {}).get("generators", ["gaussian"]))
    if "gaussian" in generators:
        rng = torch.Generator().manual_seed(0)
        z = fit_gaussian(attr_cfg, codes).sample(examples, generator=rng)
        cols.append(("Gaussian", [decode(z[i]) for i in range(examples)]))
    if "gmm" in generators:
        rng = torch.Generator().manual_seed(1)
        z = fit_gmm(attr_cfg, codes).sample(examples, generator=rng)
        cols.append(("GMM", [decode(z[i]) for i in range(examples)]))
    if "ddpm" in generators:
        ddpm = train_ddpm(attr_cfg, codes, device=device, progress=False)["model"]
        z = ddpm.sample(examples, device=device).cpu()
        cols.append(("DDPM", [decode(z[i]) for i in range(examples)]))

    ncols = len(cols)
    fig = plt.figure(figsize=(2.9 * ncols, 2.2 * examples))
    for c, (header, meshes) in enumerate(cols):
        for r, mesh in enumerate(meshes):
            ax = fig.add_subplot(examples, ncols, r * ncols + c + 1, projection="3d")
            _render_mesh(ax, mesh, header if r == 0 else None)
    # No suptitle: the caption already says which cell this is, and the headers
    # already say what the columns are.
    fig.subplots_adjust(
        left=0.005, right=0.995, top=0.94, bottom=0.005, wspace=0.0, hspace=0.0
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="outputs/grid")
    parser.add_argument("--figures", type=str, default="figures")
    parser.add_argument(
        "--gallery-cell",
        type=str,
        default=None,
        help="cell to render the gallery from, e.g. N10_D16 (default: largest N)",
    )
    parser.add_argument(
        "--gallery-examples",
        type=int,
        default=3,
        help="rows in the gallery, i.e. independent draws per category",
    )
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
    plot_gallery(
        summary,
        input_dir,
        fig_dir / "gallery.png",
        args.gallery_cell,
        args.gallery_examples,
    )
    print(f"Wrote tables and figures -> {fig_dir}")


if __name__ == "__main__":
    main()
