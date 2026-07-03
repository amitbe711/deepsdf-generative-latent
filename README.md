# Generative Latent Models for DeepSDF: Gaussian vs Diffusion in the Small-Data Regime

Final project — Deep Learning for 3D Computer Vision (Hebrew University).  
**Author:** Amit Benbenishti

DeepSDF learns one latent code per shape but not a distribution over codes. This
project trains a **DeepSDF auto-decoder** (Stage 1), freezes it, then fits two
generators on the latent codes — a **Gaussian** and a **latent DDPM** — and
compares them across dataset sizes `N ∈ {10, 50, 150}` and latent dimensions
`D ∈ {16, 32}`.

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

## Citation / related work

Built on ideas from DeepSDF (Park et al., 2019), DDPM (Ho et al., 2020), and
3D-LDM (Nam et al., 2022). All algorithmic code under `code/src/` is an original
implementation — see [`code/src/thirdparty/README.md`](code/src/thirdparty/README.md).
