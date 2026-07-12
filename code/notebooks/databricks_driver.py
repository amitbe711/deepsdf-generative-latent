# Databricks notebook source
# MAGIC %md
# MAGIC # DeepSDF generative latent — Databricks driver
# MAGIC
# MAGIC **Recommended:** run as a **Job** on an **on-demand** `g5.8xlarge` (A10G) — not spot.
# MAGIC
# MAGIC Flow:
# MAGIC 1. ShapeNet zip on **S3** (HF download once)
# MAGIC 2. Extract mesh subset to `/local_disk0`
# MAGIC 3. Preflight → `run_grid.py` → outputs + **logs** to **S3**
# MAGIC
# MAGIC Logs: `s3://.../3d_project/logs/<output_name>_<timestamp>.log`
# MAGIC
# MAGIC **Secret:** scope `hf`, key `token`

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
from datetime import datetime, timezone
from pathlib import Path

import yaml

# ── S3 layout ────────────────────────────────────────────────────────────────
S3_BASE = "s3://sw-dmi-data-staging/users/amit.benbenishti/others/3d_project"
S3_SHAPENET_ZIP = f"{S3_BASE}/data/shapenet/03001627.zip"
S3_OUTPUTS = f"{S3_BASE}/outputs"
S3_LOGS = f"{S3_BASE}/logs"

LOCAL_BASE = Path("/local_disk0/3d_project")
LOCAL_SHAPENET = LOCAL_BASE / "shapenet" / "03001627"
LOCAL_ZIP = LOCAL_BASE / "03001627.zip"
LOCAL_OUTPUTS = LOCAL_BASE / "outputs"
LOCAL_REPO = LOCAL_BASE / "deepsdf-generative-latent"
REPO_URL = "https://github.com/amitbe711/deepsdf-generative-latent.git"

# ── Run knobs (override via widgets below) ─────────────────────────────────
RUN_CONFIG = "configs/shapenet_overnight_n50.yaml"
OUTPUT_NAME = "shapenet_overnight_n50"
ONLY_D = "16"
ONLY_N = ""
MESHES_FOR_RUN = 0  # 0 = auto from config (N + reference + 10)
RECON_RESOLUTION_CAP = 40  # lower = less RAM / GPU pressure during decode

# COMMAND ----------

# Widgets (Databricks UI — optional overrides)
dbutils.widgets.dropdown(  # noqa: F821
    "run_config",
    RUN_CONFIG,
    [
        "configs/shapenet_single_n1.yaml",
        "configs/shapenet_quick_n10.yaml",
        "configs/shapenet_overnight_n50.yaml",
        "configs/shapenet_overnight_n500.yaml",
    ],
)
dbutils.widgets.text("output_name", OUTPUT_NAME)  # noqa: F821
dbutils.widgets.text("only_d", ONLY_D)  # noqa: F821
dbutils.widgets.text("only_n", ONLY_N)  # noqa: F821
dbutils.widgets.text("meshes_for_run", "0")  # noqa: F821
dbutils.widgets.text("recon_cap", str(RECON_RESOLUTION_CAP))  # noqa: F821

RUN_CONFIG = dbutils.widgets.get("run_config")  # noqa: F821
OUTPUT_NAME = dbutils.widgets.get("output_name")  # noqa: F821
ONLY_D = dbutils.widgets.get("only_d")  # noqa: F821
ONLY_N = dbutils.widgets.get("only_n")  # noqa: F821
MESHES_FOR_RUN = int(dbutils.widgets.get("meshes_for_run"))  # noqa: F821
RECON_RESOLUTION_CAP = int(dbutils.widgets.get("recon_cap"))  # noqa: F821

RUN_TS = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
LOCAL_LOG = LOCAL_BASE / "logs" / f"{OUTPUT_NAME}_{RUN_TS}.log"
S3_LOG = f"{S3_LOGS}/{OUTPUT_NAME}_{RUN_TS}.log"

# COMMAND ----------


class _Tee:
    """Mirror stdout to a local log file (flushed every line)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._terminal = sys.stdout
        self._log = path.open("w", encoding="utf-8")

    def write(self, msg: str) -> None:
        self._terminal.write(msg)
        self._log.write(msg)
        self._log.flush()

    def flush(self) -> None:
        self._terminal.flush()
        self._log.flush()

    def close(self) -> None:
        self._log.close()


def upload_log() -> None:
    if LOCAL_LOG.exists():
        dbutils.fs.mkdirs(S3_LOGS)  # noqa: F821
        dbutils.fs.cp(f"file:{LOCAL_LOG}", S3_LOG, recurse=False)  # noqa: F821
        print(f"Log uploaded -> {S3_LOG}")


def s3_cp(src: str, dst: str, recurse: bool = True) -> None:
    dbutils.fs.cp(src, dst, recurse=recurse)  # noqa: F821


def ensure_shapenet_zip_on_s3() -> None:
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
        for name in zf.namelist():
            if not name.startswith(prefixes) or name.endswith("/"):
                continue
            out = dest_dir.parent / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(zf.read(name))

    objs = list(dest_dir.rglob("model_normalized.obj"))
    print(f"Extracted {len(objs)} model_normalized.obj files to {dest_dir}")
    return len(objs)


def meshes_needed(cfg: dict) -> int:
    if MESHES_FOR_RUN > 0:
        return MESHES_FOR_RUN
    grid_n = int(ONLY_N) if ONLY_N else max(cfg["grid"]["N"])
    ref = int(cfg["eval"].get("num_reference", 50))
    return grid_n + ref + 10


def sync_meshes_to_local(limit: int) -> Path:
    LOCAL_BASE.mkdir(parents=True, exist_ok=True)
    objs = list(LOCAL_SHAPENET.rglob("model_normalized.obj")) if LOCAL_SHAPENET.exists() else []
    if len(objs) >= limit:
        print(f"Local meshes already present: {len(objs)}")
        return LOCAL_SHAPENET

    if not LOCAL_ZIP.exists():
        print("Copying zip from S3 to local disk...")
        s3_cp(S3_SHAPENET_ZIP, f"file:{LOCAL_ZIP}", recurse=False)

    n = extract_mesh_subset_from_zip(LOCAL_ZIP, LOCAL_SHAPENET, limit)
    if n < min(limit, 30):
        raise RuntimeError(f"Only {n} meshes extracted — need >= {min(limit, 30)}")
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


def apply_stability_patches(cfg: dict) -> dict:
    """Databricks-friendly defaults for real-mesh decode stability."""
    dec = cfg.setdefault("decoder", {})
    dec["use_tanh"] = False

    s1 = cfg.setdefault("stage1", {})
    s1["code_reg_lambda"] = float(s1.get("code_reg_lambda", 0.0) or 0.0)
    s1["lr_codes"] = max(float(s1.get("lr_codes", 1e-3)), 5e-3)

    ev = cfg.setdefault("eval", {})
    ev["recon_resolution"] = min(int(ev.get("recon_resolution", 48)), RECON_RESOLUTION_CAP)
    return cfg


def patch_config(config_path: Path, mesh_dir: Path, out_yaml: Path) -> Path:
    cfg = yaml.safe_load(config_path.read_text())
    cfg["data"]["source"] = "mesh_dir"
    cfg["data"]["mesh_dir"] = str(mesh_dir)
    cfg["device"] = "cuda"
    cfg = apply_stability_patches(cfg)
    out_yaml.write_text(yaml.dump(cfg))
    print("Patched config ->", out_yaml)
    print("  recon_resolution:", cfg["eval"]["recon_resolution"])
    print("  use_tanh:", cfg["decoder"]["use_tanh"])
    return out_yaml


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


def run_grid_logged(code_dir: Path, config_path: Path, output_dir: Path) -> None:
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
    print("Running:", " ".join(cmd), flush=True)

    proc = subprocess.Popen(
        cmd,
        cwd=str(code_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def upload_outputs(local_output: Path, s3_output: str) -> None:
    if not local_output.exists():
        print("No local outputs to upload:", local_output)
        return
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
    else:
        print("No summary.json — run may have been interrupted.")
    print("S3 outputs:", s3_out)
    print("S3 log:", S3_LOG)

# COMMAND ----------

tee = _Tee(LOCAL_LOG)
sys.stdout = tee
print(f"Run: {OUTPUT_NAME}  config: {RUN_CONFIG}  log: {S3_LOG}")

local_out = LOCAL_OUTPUTS / OUTPUT_NAME
s3_out = f"{S3_OUTPUTS}/{OUTPUT_NAME}"
run_yaml = LOCAL_BASE / "run_config.yaml"

try:
    import torch
    print("CUDA:", torch.cuda.is_available(), end=" ")
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    else:
        print("(no GPU — use a GPU cluster)")

    ensure_shapenet_zip_on_s3()
    upload_log()

    code_dir = setup_repo()
    cfg = yaml.safe_load((code_dir / RUN_CONFIG).read_text())
    mesh_limit = meshes_needed(cfg)
    print(f"Meshes to extract: {mesh_limit}")

    mesh_dir = sync_meshes_to_local(mesh_limit)
    upload_log()

    run_preflight(code_dir, mesh_dir)
    upload_log()

    LOCAL_OUTPUTS.mkdir(parents=True, exist_ok=True)
    patch_config(code_dir / RUN_CONFIG, mesh_dir, run_yaml)
    run_grid_logged(code_dir, run_yaml, local_out)
    upload_log()

    upload_outputs(local_out, s3_out)
    print_metrics(local_out, s3_out)

except Exception as exc:
    print(f"RUN FAILED: {exc}", flush=True)
    raise
finally:
    sys.stdout = tee._terminal
    tee.close()
    upload_log()
