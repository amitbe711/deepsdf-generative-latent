# Databricks notebook source
# MAGIC %md
# MAGIC # DeepSDF generative latent — Databricks driver
# MAGIC
# MAGIC Same flow as Colab:
# MAGIC 1. Ensure ShapeNet chairs on **S3** (download from Hugging Face once)
# MAGIC 2. Extract a subset from zip on **local disk** (`/local_disk0`)
# MAGIC 3. **Preflight** mesh load + SDF + decode smoke test
# MAGIC 4. Clone repo → `run_grid.py` → save outputs back to **S3**
# MAGIC
# MAGIC **Cluster:** single-node GPU (e.g. `g4dn.xlarge` or better), ML runtime 14.x+.
# MAGIC **Secret:** scope `hf`, key `token` → Hugging Face Read token.

# COMMAND ----------

# MAGIC %pip install -q huggingface_hub rtree charset-normalizer

# COMMAND ----------

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import yaml

# ── S3 layout ────────────────────────────────────────────────────────────────
S3_BASE = "s3://sw-dmi-data-staging/users/amit.benbenishti/others/3d_project"
S3_SHAPENET_ZIP = f"{S3_BASE}/data/shapenet/03001627.zip"
S3_SHAPENET_DIR = f"{S3_BASE}/data/shapenet/03001627"
S3_OUTPUTS = f"{S3_BASE}/outputs"

LOCAL_BASE = Path("/local_disk0/3d_project")
LOCAL_SHAPENET = LOCAL_BASE / "shapenet" / "03001627"
LOCAL_ZIP = LOCAL_BASE / "03001627.zip"
LOCAL_OUTPUTS = LOCAL_BASE / "outputs"
LOCAL_REPO = LOCAL_BASE / "deepsdf-generative-latent"

RUN_CONFIG = "configs/shapenet_quick_n10.yaml"
OUTPUT_NAME = "shapenet_quick_n10"
ONLY_D = "16"
ONLY_N = ""
MESHES_FOR_RUN = 60
REPO_URL = "https://github.com/amitbe711/deepsdf-generative-latent.git"

# COMMAND ----------


def s3_count_entries(path: str) -> int:
    try:
        return len(dbutils.fs.ls(path))  # noqa: F821
    except Exception:
        return 0


def s3_cp(src: str, dst: str, recurse: bool = True) -> None:
    dbutils.fs.cp(src, dst, recurse=recurse)  # noqa: F821


def ensure_shapenet_zip_on_s3() -> None:
    """Persist 03001627.zip on S3 (extracted tree optional; zip is enough)."""
    try:
        dbutils.fs.ls(S3_SHAPENET_ZIP)  # noqa: F821
        print("Zip already on S3:", S3_SHAPENET_ZIP)
        return
    except Exception:
        pass

    from huggingface_hub import hf_hub_download, login

    try:
        login(token=dbutils.secrets.get(scope="hf", key="token"))  # noqa: F821
    except Exception:
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
    dbutils.fs.mkdirs(f"{S3_BASE}/data/shapenet")  # noqa: F821
    s3_cp(f"file:{zip_path}", S3_SHAPENET_ZIP, recurse=False)
    print("Uploaded zip to", S3_SHAPENET_ZIP)
    shutil.rmtree(tmp, ignore_errors=True)


def extract_mesh_subset_from_zip(zip_path: Path, dest_dir: Path, limit: int) -> int:
    """Extract first ``limit`` ShapeNet chair models from zip into dest_dir."""
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        model_ids: list[str] = []
        for name in zf.namelist():
            if name.endswith("model_normalized.obj"):
                parts = Path(name).parts
                if len(parts) >= 4 and parts[0] == "03001627":
                    model_ids.append(parts[1])
        model_ids = sorted(set(model_ids))[:limit]
        prefixes = tuple(f"03001627/{mid}/" for mid in model_ids)
        written = 0
        for name in zf.namelist():
            if not name.startswith(prefixes):
                continue
            if name.endswith("/"):
                continue
            out = dest_dir.parent / name  # .../shapenet/03001627/<id>/...
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(zf.read(name))
            written += 1
    objs = list(dest_dir.rglob("model_normalized.obj"))
    print(f"Extracted {len(objs)} model_normalized.obj files to {dest_dir}")
    return len(objs)


def sync_meshes_to_local(limit: int) -> Path:
    """Pull zip from S3 once, extract ``limit`` models to local NVMe."""
    LOCAL_BASE.mkdir(parents=True, exist_ok=True)
    objs = list(LOCAL_SHAPENET.rglob("model_normalized.obj")) if LOCAL_SHAPENET.exists() else []
    if len(objs) >= limit:
        print(f"Local meshes already present: {len(objs)}")
        return LOCAL_SHAPENET

    if not LOCAL_ZIP.exists():
        print("Copying zip from S3 to local disk...")
        s3_cp(S3_SHAPENET_ZIP, f"file:{LOCAL_ZIP}", recurse=False)

    n = extract_mesh_subset_from_zip(LOCAL_ZIP, LOCAL_SHAPENET, limit)
    if n < 30:
        raise RuntimeError(
            f"Only {n} meshes extracted — expected >= 30 for N=10 + reference. "
            "Check zip integrity on S3."
        )
    return LOCAL_SHAPENET


def setup_repo() -> Path:
    code_dir = LOCAL_REPO / "code"
    if not (code_dir / "scripts" / "run_grid.py").exists():
        if LOCAL_REPO.exists():
            shutil.rmtree(LOCAL_REPO)
        subprocess.check_call(["git", "clone", REPO_URL, str(LOCAL_REPO)])
    else:
        subprocess.check_call(["git", "-C", str(LOCAL_REPO), "pull"])
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(code_dir / "requirements.txt")]
    )
    return code_dir


def run_preflight(code_dir: Path, mesh_dir: Path) -> None:
    subprocess.check_call(
        [
            sys.executable,
            "scripts/preflight_mesh.py",
            "--mesh-dir",
            str(mesh_dir),
            "--code-dir",
            str(code_dir),
            "--device",
            "cuda",
        ],
        cwd=str(code_dir),
    )


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


def print_metrics(local_out: Path, s3_out: str) -> None:
    summary = local_out / "summary.json"
    if summary.exists():
        rows = json.loads(summary.read_text())
        for row in rows:
            print(f"N={row['N']} D={row['D']} recon={row['reconstruction']}")
            hist = row.get("stage1_history", [])
            if hist:
                last = hist[-1]
                print(f"  stage1 last: step={last['step']} loss={last['loss']:.4f}")
            for name, g in row.get("generators", {}).items():
                print(
                    f"  {name}: cov={g['coverage']:.3f} mmd={g['mmd']:.4f} "
                    f"1nn={g['one_nn_acc']:.3f} valid={g['valid_ratio']:.2f}"
                )
    print("S3 outputs:", s3_out)


# COMMAND ----------

ensure_shapenet_zip_on_s3()

# COMMAND ----------

mesh_dir = sync_meshes_to_local(MESHES_FOR_RUN)

# COMMAND ----------

code_dir = setup_repo()
run_preflight(code_dir, mesh_dir)

# COMMAND ----------

LOCAL_OUTPUTS.mkdir(parents=True, exist_ok=True)
local_out = LOCAL_OUTPUTS / OUTPUT_NAME
s3_out = f"{S3_OUTPUTS}/{OUTPUT_NAME}"

run_yaml = patch_config(code_dir / RUN_CONFIG, mesh_dir, LOCAL_BASE / "run_config.yaml")
run_grid(code_dir, run_yaml, local_out)

# COMMAND ----------

upload_outputs(local_out, s3_out)
print_metrics(local_out, s3_out)
