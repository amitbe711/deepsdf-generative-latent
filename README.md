# Generative Latent Models for DeepSDF: Gaussian vs Diffusion in the Small-Data Regime

Final project — Deep Learning for 3D Computer Vision (Hebrew University).  
**Author:** Amit Benbenishti

DeepSDF learns one latent code per shape but not a distribution over codes. This
project trains a **DeepSDF auto-decoder** (Stage 1), freezes it, then fits three
generators on the latent codes — a **Gaussian**, a **GMM** (EM, from scratch), and
a **latent DDPM** — and compares them as the dataset shrinks,
`N ∈ {10, 50, 150}`, on ShapeNet chairs. Latent dimension is held at `D = 16` for
the sweep and ablated separately at `D = 32`; see the report for why.

## Repository layout

| Path | Contents |
|------|----------|
| [`code/`](code/) | All training/evaluation code, configs, tests, Colab notebook |
| [`report/`](report/) | LaTeX report source and generated figures |
| [`proposal.md`](proposal.md) | Approved project proposal |

See [`code/README.md`](code/README.md) for install, reproduction, and data setup.

## Quick start

```bash
cd code
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Fast CPU smoke test (~minutes)
python scripts/run_grid.py --config configs/smoke.yaml --output outputs/smoke
python scripts/make_figures.py --input outputs/smoke --figures figures_smoke
```

## Reproducing the reported results

The report's numbers come from ShapeNet chairs (synset `03001627`) on a GPU;
[`code/notebooks/colab_driver.ipynb`](code/notebooks/colab_driver.ipynb) runs the
whole thing on Colab, including the mesh download. Equivalent CLI, given a chair
mesh directory configured in `configs/shapenet_grid.yaml`:

```bash
cd code
OUT=outputs/shapenet_scoped
# Main sweep: N = 10, 50, 150 at D = 16
python scripts/run_grid.py --config configs/shapenet_grid.yaml --output $OUT --only-D 16
# Latent-dimension ablation: one D = 32 cell at N = 50
python scripts/run_grid.py --config configs/shapenet_grid.yaml --output $OUT --only-N 50 --only-D 32
# Tables + figures for the report
python scripts/make_figures.py --input $OUT --figures ../report/figures --gallery-cell N10_D16
```

Both runs accumulate into one `$OUT/summary.json`, and cells that already have a
`metrics.json` are skipped, so an interrupted run resumes by re-issuing the same
command (`--force` re-runs anyway). End to end the four cells take roughly 2.5
hours on a Colab T4, most of it SDF-sampling meshes off Drive rather than
training.

### Headline result

The GMM prior gives the best Coverage at every N (0.325 / 0.300 / 0.400), the
Gaussian trails it closely, and the DDPM wins no cell: it is the worst generator
on every generation metric at N=150, and at N=10 only 15% of its samples decode to
a surface (against 100% for the Gaussian). 1-NN accuracy stays in 0.900–0.988
against an ideal of 0.5, so all three priors remain trivially separable from real
chairs — the ranking is between degrees of failure. See `report/main.pdf`.

Build the PDF with `cd report && tectonic main.tex` (see
[`report/README.md`](report/README.md)), then package the submission:

```bash
cd code && python scripts/make_submission.py --name AmitBenbenishti_final
```

## Citation / related work

Built on ideas from DeepSDF (Park et al., 2019), DDPM (Ho et al., 2020), and
3D-LDM (Nam et al., 2022). All algorithmic code under `code/src/` is an original
implementation — see [`code/src/thirdparty/README.md`](code/src/thirdparty/README.md).
