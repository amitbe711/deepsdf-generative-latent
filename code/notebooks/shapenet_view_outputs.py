# Databricks notebook source
# MAGIC %md
# MAGIC # ShapeNet — view metrics & shape gallery
# MAGIC
# MAGIC Run **after** a grid job finishes. Reads `summary.json` + checkpoint from S3,
# MAGIC plots metrics tables/curves, and decodes example chairs (GT / recon / samples).
# MAGIC
# MAGIC **S3 base:** `s3://sw-dmi-data-staging/users/amit.benbenishti/others/3d_project`

# COMMAND ----------

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import trimesh

S3_BASE = "s3://sw-dmi-data-staging/users/amit.benbenishti/others/3d_project"
LOCAL_BASE = Path("/local_disk0/3d_project")
REPO = LOCAL_BASE / "deepsdf-generative-latent" / "code"
MESH_DIR = LOCAL_BASE / "shapenet" / "03001627"

dbutils.widgets.text("output_name", "shapenet_quick_n10")  # noqa: F821
dbutils.widgets.text("cell_tag", "N10_D16")  # noqa: F821
dbutils.widgets.text("decode_res", "48")  # noqa: F821
dbutils.widgets.text("num_examples", "4")  # noqa: F821

OUTPUT_NAME = dbutils.widgets.get("output_name")  # noqa: F821
CELL_TAG = dbutils.widgets.get("cell_tag")  # noqa: F821
DECODE_RES = int(dbutils.widgets.get("decode_res"))  # noqa: F821
NUM_EXAMPLES = int(dbutils.widgets.get("num_examples"))  # noqa: F821

S3_OUT = f"{S3_BASE}/outputs/{OUTPUT_NAME}"
LOCAL_VIEW = LOCAL_BASE / "view" / OUTPUT_NAME

# COMMAND ----------

# Sync outputs from S3
if LOCAL_VIEW.exists():
    import shutil
    shutil.rmtree(LOCAL_VIEW)
LOCAL_VIEW.mkdir(parents=True)

print("Syncing", S3_OUT)
for f in dbutils.fs.ls(S3_OUT):  # noqa: F821
    print(" ", f.path)
dbutils.fs.cp(S3_OUT, f"file:{LOCAL_VIEW}", recurse=True)  # noqa: F821

summary_path = LOCAL_VIEW / "summary.json"
if not summary_path.exists():
    raise FileNotFoundError(f"No summary.json in {LOCAL_VIEW}")

summary = json.loads(summary_path.read_text())
print(f"Loaded {len(summary)} grid cell(s)")

# COMMAND ----------

# Metrics table
rows = []
for cell in summary:
    recon = cell.get("reconstruction", {})
    for gen_name, g in cell.get("generators", {}).items():
        rows.append({
            "N": cell["N"],
            "D": cell["D"],
            "generator": gen_name,
            "recon_chamfer": recon.get("chamfer"),
            "recon_iou": recon.get("iou"),
            "coverage": g.get("coverage"),
            "mmd": g.get("mmd"),
            "one_nn": g.get("one_nn_acc"),
            "valid": g.get("valid_ratio"),
            "runtime_min": round(cell.get("seconds", 0) / 60, 1),
        })

df = pd.DataFrame(rows)
display(df)  # noqa: F821

# COMMAND ----------

# Loss curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for cell in summary:
    tag = f"N{cell['N']}_D{cell['D']}"
    s1 = cell.get("stage1_history", [])
    if s1:
        axes[0].plot([h["step"] for h in s1], [h["loss"] for h in s1], label=tag)
    ddpm = cell.get("ddpm_history", [])
    if ddpm:
        axes[1].plot([h["step"] for h in ddpm], [h["loss"] for h in ddpm], label=tag)
axes[0].set_title("Stage-1 loss"); axes[0].set_xlabel("step"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].set_title("DDPM loss"); axes[1].set_xlabel("step"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
display(fig)  # noqa: F821

# COMMAND ----------

# Generation metric bars (one subplot per metric)
if df.empty:
    raise RuntimeError("No generator rows in summary")

metrics = [
    ("coverage", "Coverage (higher)"),
    ("mmd", "MMD (lower)"),
    ("one_nn", "1-NN (0.5 ideal)"),
]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (col, title) in zip(axes, metrics):
    labels = [f"{r.generator}\nN={r.N} D={r.D}" for r in df.itertuples()]
    ax.bar(range(len(df)), df[col])
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    if col == "one_nn":
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
fig.suptitle(OUTPUT_NAME)
plt.tight_layout()
display(fig)  # noqa: F821

# COMMAND ----------

# Shape gallery: GT | recon | Gaussian | GMM | DDPM
if not (REPO / "src" / "sample.py").exists():
    subprocess.check_call([
        "git", "clone", "https://github.com/amitbe711/deepsdf-generative-latent.git",
        str(REPO.parent),
    ])
sys.path.insert(0, str(REPO))

from src.data.dataset import _load_meshes_from_dir
from src.data.normalize import normalize_mesh_to_unit_sphere
from src.data.sdf_sampling import sample_surface_points
from src.models.decoder import DeepSDFDecoder
from src.sample import decode_point_cloud
from src.train import fit_gaussian, fit_gmm, train_ddpm
from src.utils import load_checkpoint
from src.utils.config import AttrDict

ckpt_path = LOCAL_VIEW / CELL_TAG / "checkpoint.pt"
if not ckpt_path.exists():
    available = [p.name for p in LOCAL_VIEW.iterdir() if p.is_dir()]
    raise FileNotFoundError(f"{ckpt_path} not found. Available cells: {available}")

device = "cuda" if torch.cuda.is_available() else "cpu"
decode_device = "cpu" if device == "cuda" else device
print(f"device={device}  decode={decode_device}")

ckpt = load_checkpoint(ckpt_path)
cfg = AttrDict(ckpt["config"])
codes = ckpt["codes_state"]["embedding.weight"]
latent_dim = ckpt["latent_dim"]

decoder = DeepSDFDecoder(
    latent_dim=latent_dim,
    hidden_dim=int(cfg.decoder.hidden_dim),
    num_layers=int(cfg.decoder.num_layers),
    skip_in=tuple(cfg.decoder.skip_in),
    dropout_prob=float(cfg.decoder.dropout_prob),
    use_weight_norm=bool(cfg.decoder.use_weight_norm),
    use_tanh=bool(cfg.decoder.use_tanh),
    geometric_init=bool(cfg.decoder.get("geometric_init", True)),
    init_radius=float(cfg.decoder.get("init_radius", 0.5)),
)
decoder.load_state_dict(ckpt["decoder_state"])
decoder.eval()

n_pts = 2000
gen = torch.Generator().manual_seed(0)

def mesh_pc(mesh: trimesh.Trimesh) -> np.ndarray:
    return sample_surface_points(normalize_mesh_to_unit_sphere(mesh), n_pts)

def decode_pc(z: torch.Tensor):
    return decode_point_cloud(
        decoder, z.detach().cpu(), num_points=n_pts,
        resolution=DECODE_RES, device=decode_device,
    )

def scatter3d(ax, pc, title: str) -> None:
    if pc is None:
        ax.text2D(0.25, 0.5, "decode failed", transform=ax.transAxes)
    else:
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=0.3, alpha=0.6)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_axis_off()

gt_meshes = _load_meshes_from_dir(MESH_DIR, limit=NUM_EXAMPLES) if MESH_DIR.exists() else []
prior_g = fit_gaussian(cfg, codes)
prior_m = fit_gmm(cfg, codes)
print("Training DDPM for gallery samples (~10s)...")
ddpm = train_ddpm(cfg, codes, device=device, progress=False, verbose=True)["model"]

z_g = prior_g.sample(NUM_EXAMPLES, generator=gen)
z_m = prior_m.sample(NUM_EXAMPLES, generator=gen)
z_d = ddpm.sample(NUM_EXAMPLES, device=device).cpu()

gallery_rows = []
if gt_meshes:
    gallery_rows.append(("GT (ShapeNet)", [mesh_pc(m) for m in gt_meshes]))
gallery_rows += [
    ("Reconstruction", [decode_pc(codes[i]) for i in range(min(NUM_EXAMPLES, codes.shape[0]))]),
    ("Gaussian", [decode_pc(z_g[i]) for i in range(NUM_EXAMPLES)]),
    ("GMM", [decode_pc(z_m[i]) for i in range(NUM_EXAMPLES)]),
    ("DDPM", [decode_pc(z_d[i]) for i in range(NUM_EXAMPLES)]),
]

fig = plt.figure(figsize=(3.2 * NUM_EXAMPLES, 3.2 * len(gallery_rows)))
for r, (row_title, clouds) in enumerate(gallery_rows):
    for c, pc in enumerate(clouds):
        ax = fig.add_subplot(len(gallery_rows), NUM_EXAMPLES, r * NUM_EXAMPLES + c + 1, projection="3d")
        scatter3d(ax, pc, f"{row_title} {c}")

fig.suptitle(f"{OUTPUT_NAME} / {CELL_TAG}  decode_res={DECODE_RES}", y=1.01)
plt.tight_layout()
display(fig)  # noqa: F821

gallery_png = LOCAL_VIEW / f"gallery_{CELL_TAG}.png"
fig.savefig(gallery_png, dpi=150, bbox_inches="tight")
dbutils.fs.cp(f"file:{gallery_png}", f"{S3_OUT}/", recurse=False)  # noqa: F821
print("Gallery saved:", gallery_png)
print("S3:", f"{S3_OUT}/gallery_{CELL_TAG}.png")

# COMMAND ----------

# Regenerate report figures locally (optional — copy to report/figures/)
fig_dir = LOCAL_VIEW / "figures"
subprocess.check_call([
    sys.executable, "scripts/make_figures.py",
    "--input", str(LOCAL_VIEW),
    "--figures", str(fig_dir),
], cwd=str(REPO))

print("Report figures ->", fig_dir)
for p in sorted(fig_dir.glob("*")):
    print(" ", p.name, p.stat().st_size)
