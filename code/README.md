# Generative Latent Models for DeepSDF: Gaussian vs Diffusion in the Small-Data Regime

Code for the final project. We train a **DeepSDF auto-decoder** (Stage 1), then
fit two generators over the frozen latent codes (Stage 2) - a **Gaussian** and a
**latent DDPM** - and compare them across a grid of dataset sizes `N in {10, 50,
150}` and latent dimensions `D in {16, 32}`. The question is *when the extra
complexity of diffusion is worth it over a simple Gaussian in the low-data,
low-capacity regime*.

## Pipeline

```
meshes ──▶ SDF samples ──▶ [Stage 1] DeepSDF auto-decoder f_theta(z, x)
                                         │  (joint theta + per-shape codes z_i)
                                         ▼
                              frozen decoder + codes {z_i}
                                         │
              ┌──────────────────────────┴──────────────────────────┐
        [Stage 2] Gaussian(mean, cov)                     [Stage 2] latent DDPM
              └──────────────────────────┬──────────────────────────┘
                                         ▼
            sample z ──▶ decode f_theta(z, grid) ──▶ Marching Cubes ──▶ mesh
```

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

On Google Colab, dependencies (except `trimesh`/`rtree`) are preinstalled; run
`!pip install -r requirements.txt` in the driver notebook.

## Reproduce the results

Everything is driven by two scripts and a YAML config.

```bash
# 1. Run the full N x D grid (Stage 1 + both generators per cell).
python scripts/run_grid.py --config configs/base.yaml  --output outputs/grid

# 2. Regenerate every table and figure used in the report.
python scripts/make_figures.py --input outputs/grid --figures figures
```

For a fast end-to-end check on CPU (used by CI / local dev), swap in the smoke
config, which uses the identical code path with tiny counts:

```bash
python scripts/run_grid.py     --config configs/smoke.yaml --output outputs/smoke
python scripts/make_figures.py --input outputs/smoke       --figures figures_smoke
```

Outputs per grid cell (`outputs/<run>/N{N}_D{D}/`):
- `checkpoint.pt` - decoder weights + latent codes + config (resumable),
- `metrics.json`  - reconstruction and generation metrics + loss histories.

`make_figures.py` writes `results.csv`, `results_table.tex`, and the
`degradation_*`, `loss_curves`, and `gallery` PNGs.

## Data

- **`synthetic`** (default): analytic parametric "chairs" with exact SDFs - no
  preprocessing, fully reproducible, good for development and the low-data study.
- **`mesh_dir`**: point `data.mesh_dir` at a directory of meshes (e.g. a ShapeNet
  chairs subset). Meshes are normalized into the unit sphere and SDF-sampled with
  the trimesh proximity query. Inspect a directory with
  `python scripts/prepare_data.py --inspect-dir /path/to/chairs`.

### ShapeNet chairs (synset `03001627`)

1. Register at [shapenet.org](https://shapenet.org/) (or accept the license on
   Hugging Face).
2. Download the chairs category and unzip so you have paths like
   `03001627/<model_id>/models/model_normalized.obj`.
3. Edit `configs/shapenet.yaml` → set `data.mesh_dir` to that folder.
4. Quick sanity check (1 mesh): `python scripts/run_grid.py --config configs/shapenet_smoke.yaml --output outputs/shapenet_smoke`
5. Full grid: `python scripts/run_grid.py --config configs/shapenet.yaml --output outputs/shapenet`

**Hugging Face** (after login): `hf auth login`, then download
`ShapeNet/ShapeNetCore` file `03001627.zip` and unzip. Training uses the first
`N` meshes; generation metrics use the next `num_reference` meshes as a held-out
set (falls back to synthetic chairs if there are not enough).

### Databricks (GPU job)

Import `notebooks/databricks_driver.py` as a Databricks notebook. It mirrors the Colab flow:

1. Download chairs from Hugging Face **once** → persist zip on S3
2. Extract `N + reference + 10` meshes to `/local_disk0`
3. Preflight → `run_grid.py` → upload **outputs + logs** to S3

**Cluster:** on-demand `g5.8xlarge` (A10G) recommended; avoid spot for long runs.
**Job:** run as a Databricks Job so cluster restarts / disconnects are less likely.

Default S3 prefix:

```
s3://sw-dmi-data-staging/users/amit.benbenishti/others/3d_project/
  data/shapenet/03001627.zip
  outputs/shapenet_overnight_n50/
  logs/shapenet_overnight_n50_<timestamp>.log
```

Notebook widgets: `run_config`, `output_name`, `only_d`, `meshes_for_run`, `recon_cap`.

## Metrics

- **Reconstruction:** Chamfer distance, volumetric IoU.
- **Generation** (Achlioptas et al., 2018): Coverage, MMD-CD, 1-NN accuracy
  (0.5 is ideal) against a held-out reference set of real shapes.

## Layout

```
configs/       base.yaml (full grid) + smoke.yaml (fast CPU)
src/
  data/        mesh normalization, SDF sampling, synthetic shapes, dataset
  models/      decoder, latent codes, gaussian prior, latent DDPM
  train/       stage-1 auto-decoder, stage-2 gaussian, stage-2 DDPM
  metrics/     reconstruction (chamfer, iou), generation (coverage, mmd, 1-nn)
  marching.py  SDF grid -> mesh (Marching Cubes)
  sample.py    latent code -> SDF volume -> mesh / point cloud
  thirdparty/  (empty) see its README for the own-vs-borrowed accounting
scripts/       run_grid.py, make_figures.py, prepare_data.py
notebooks/     colab_driver.ipynb, databricks_driver.py
tests/         sanity checks (overfit, ddpm, metrics, sdf sampling)
```

## Authorship

All code under `src/` is the author's own implementation. Standard libraries are
used as dependencies; no external source is vendored. See
[`src/thirdparty/README.md`](src/thirdparty/README.md) for the full accounting.
