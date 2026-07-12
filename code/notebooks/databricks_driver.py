# Databricks notebook source
# MAGIC %md
# MAGIC # DeepSDF generative latent — Databricks driver
# MAGIC
# MAGIC Same flow as Colab:
# MAGIC 1. Ensure ShapeNet chairs on **S3** (download from Hugging Face once)
# MAGIC 2. Copy a subset to **local disk** (`/local_disk0`) for fast SDF sampling
# MAGIC 3. Clone repo → `run_grid.py` → save outputs back to **S3**
# MAGIC
# MAGIC **Cluster:** single-node GPU (e.g. `g4dn.xlarge` or better), ML runtime 14.x+.
# MAGIC
# MAGIC **Secret (recommended):** scope `hf`, key `token` → Hugging Face Read token.

# COMMAND ----------

# MAGIC %pip install -q huggingface_hub

# COMMAND ----------

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import yaml

# ── S3 layout (edit user prefix if needed) ───────────────────────────────────
S3_BASE = "s3://sw-dmi-data-staging/users/amit.benbenishti/others/3d_project"
S3_SHAPENET_ZIP = f"{S3_BASE}/data/shapenet/03001627.zip"
S3_SHAPENET_DIR = f"{S3_BASE}/data/shapenet/03001627"
S3_OUTPUTS = f"{S3_BASE}/outputs"

# Fast local scratch (ephemeral per cluster; re-sync from S3 each job)
LOCAL_BASE = Path("/local_disk0/3d_project")
LOCAL_SHAPENET = LOCAL_BASE / "shapenet" / "03001627"
LOCAL_OUTPUTS = LOCAL_BASE / "outputs"
LOCAL_REPO = LOCAL_BASE / "deepsdf-generative-latent"

# ── Run knobs (or use Databricks widgets below) ──────────────────────────────
RUN_CONFIG = "configs/shapenet_quick_n10.yaml"   # or shapenet_overnight_n50.yaml
OUTPUT_NAME = "shapenet_quick_n10"               # subfolder under S3_OUTPUTS
ONLY_D = "16"                                    # "" for both D in config
ONLY_N = ""                                      # e.g. "50" to override grid N
MESHES_FOR_RUN = 60                              # copy this many chairs to local disk
REPO_URL = "https://github.com/amitbe711/deepsdf-generative-latent.git"

# COMMAND ----------

# Optional widgets (uncomment on Databricks)
# dbutils.widgets.dropdown("run_config", "configs/shapenet_quick_n10.yaml",
#                          ["configs/shapenet_quick_n10.yaml", "configs/shapenet_overnight_n50.yaml"])
# dbutils.widgets.text("output_name", "shapenet_quick_n10")
# dbutils.widgets.text("only_d", "16")
# dbutils.widgets.text("meshes_for_run", "60")
# RUN_CONFIG = dbutils.widgets.get("run_config")
# OUTPUT_NAME = dbutils.widgets.get("output_name")
# ONLY_D = dbutils.widgets.get("only_d")
# MESHES_FOR_RUN = int(dbutils.widgets.get("meshes_for_run"))

# COMMAND ----------


def s3_exists(path: str) -> bool:
    try:
        dbutils.fs.ls(path)  # noqa: F821
        return True
    except Exception:
        return False


def s3_count_entries(path: str) -> int:
    try:
        return len(dbutils.fs.ls(path))  # noqa: F821
    except Exception:
        return 0


def s3_cp(src: str, dst: str, recurse: bool = True) -> None:
    dbutils.fs.cp(src, dst, recurse=recurse)  # noqa: F821


def ensure_shapenet_on_s3() -> None:
    """Download ShapeNet chairs from HF once, persist zip + extracted tree on S3."""
    if s3_count_entries(S3_SHAPENET_DIR) > 100:
        print(f"ShapeNet already on S3 ({S3_SHAPENET_DIR})")
        return

    from huggingface_hub import hf_hub_download, login

    try:
        token = dbutils.secrets.get(scope="hf", key="token")  # noqa: F821
        login(token=token)
    except Exception:
        print("No secret hf/token — calling login() interactively")
        login()

    tmp = Path(tempfile.mkdtemp(prefix="shapenet_dl_"))
    print("Downloading 03001627.zip from Hugging Face...")
    zip_path = Path(
        hf_hub_download(
            repo_id="ShapeNet/ShapeNetCore",
            filename="03001627.zip",
            repo_type="dataset",
            local_dir=str(tmp),
        )
    )
    print("Uploading zip to S3...")
    dbutils.fs.mkdirs(f"{S3_BASE}/data/shapenet")  # noqa: F821
    s3_cp(f"file:{zip_path}", S3_SHAPENET_ZIP, recurse=False)

    extract_dir = tmp / "extracted" / "03001627"
    extract_dir.mkdir(parents=True, exist_ok=True)
    print("Unzipping locally...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir.parent)

    print("Uploading extracted chairs to S3 (may take 30-60 min)...")
    s3_cp(f"file:{extract_dir}", S3_SHAPENET_DIR, recurse=True)
    print("ShapeNet persisted to", S3_SHAPENET_DIR)
    shutil.rmtree(tmp, ignore_errors=True)


def sync_meshes_to_local(limit: int) -> Path:
    """Copy first ``limit`` model folders from S3 to local NVMe for fast SDF sampling."""
    LOCAL_SHAPENET.mkdir(parents=True, exist_ok=True)
    existing = list(LOCAL_SHAPENET.iterdir()) if LOCAL_SHAPENET.exists() else []
    if len(existing) >= limit:
        print(f"Local meshes already present: {len(existing)}")
        return LOCAL_SHAPENET

    if existing:
        shutil.rmtree(LOCAL_SHAPENET)
        LOCAL_SHAPENET.mkdir(parents=True)

    entries = dbutils.fs.ls(S3_SHAPENET_DIR)  # noqa: F821
    copied = 0
    for entry in entries:
        if copied >= limit:
            break
        name = entry.name.rstrip("/")
        if name.startswith("."):
            continue
        src = f"{S3_SHAPENET_DIR}/{name}"
        dst = f"file:{LOCAL_SHAPENET / name}"
        s3_cp(src, dst, recurse=True)
        copied += 1
        if copied % 10 == 0:
            print(f"  copied {copied}/{limit}")

    print(f"Local meshes ready: {copied} folders in {LOCAL_SHAPENET}")
    return LOCAL_SHAPENET


def setup_repo() -> Path:
    code_dir = LOCAL_REPO / "code"
    if not (code_dir / "scripts" / "run_grid.py").exists():
        if LOCAL_REPO.exists():
            shutil.rmtree(LOCAL_REPO)
        print("Cloning repo...")
        subprocess.check_call(["git", "clone", REPO_URL, str(LOCAL_REPO)])
    else:
        print("Repo present, pulling...")
        subprocess.check_call(["git", "-C", str(LOCAL_REPO), "pull"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(code_dir / "requirements.txt")])
    return code_dir


def patch_config(config_path: Path, mesh_dir: Path, out_yaml: Path) -> Path:
    cfg = yaml.safe_load(config_path.read_text())
    cfg["data"]["source"] = "mesh_dir"
    cfg["data"]["mesh_dir"] = str(mesh_dir)
    cfg["device"] = "cuda"
    out_yaml.write_text(yaml.dump(cfg))
    return out_yaml


def run_grid(code_dir: Path, config_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/run_grid.py",
        "--config",
        str(config_path),
        "--output",
        str(output_dir),
    ]
    if ONLY_D:
        cmd += ["--only-D", ONLY_D]
    if ONLY_N:
        cmd += ["--only-N", ONLY_N]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(code_dir))


def upload_outputs(local_output: Path, s3_output: str) -> None:
    print("Uploading outputs to", s3_output)
    dbutils.fs.mkdirs(s3_output)  # noqa: F821
    s3_cp(f"file:{local_output}", s3_output, recurse=True)
    print("Done.")


# COMMAND ----------

ensure_shapenet_on_s3()

# COMMAND ----------

mesh_dir = sync_meshes_to_local(MESHES_FOR_RUN)

# COMMAND ----------

code_dir = setup_repo()
LOCAL_OUTPUTS.mkdir(parents=True, exist_ok=True)
local_out = LOCAL_OUTPUTS / OUTPUT_NAME
s3_out = f"{S3_OUTPUTS}/{OUTPUT_NAME}"

run_yaml = patch_config(
    code_dir / RUN_CONFIG,
    mesh_dir,
    LOCAL_BASE / "run_config.yaml",
)

run_grid(code_dir, run_yaml, local_out)

# COMMAND ----------

upload_outputs(local_out, s3_out)

# COMMAND ----------

# Quick metrics peek
import json

summary = local_out / "summary.json"
if summary.exists():
    rows = json.loads(summary.read_text())
    for row in rows:
        print(f"N={row['N']} D={row['D']} recon={row['reconstruction']}")
        for name, g in row.get("generators", {}).items():
            print(
                f"  {name}: cov={g['coverage']:.3f} mmd={g['mmd']:.4f} "
                f"1nn={g['one_nn_acc']:.3f} valid={g['valid_ratio']:.2f}"
            )
print("S3 outputs:", s3_out)
