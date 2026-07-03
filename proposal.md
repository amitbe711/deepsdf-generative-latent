Project Proposal -
Generative Latent Models for DeepSDF: Gaussian vs Diffusion in the Small-Data Regime
Author: Amit Benbenishti


Problem definition
DeepSDF learns one latent code per shape, but does not learn a distribution over codes — random codes decode to invalid shapes. I train a generative model over the latent space (not over shapes directly) and study which simple prior works when data and capacity are limited.

Approach
Stage 1: Train a DeepSDF auto-decoder on a shape collection; collect latent codes {z_i}.
Stage 2 (frozen decoder): Fit two generators on {z_i} — (1) Gaussian (mean + covariance), (2) DDPM diffusion over z (same noise-prediction setup as HW3, applied to vectors).
Inference: sample z from a generator → decode with f(z, x) → SDF → mesh.
Sweep N ∈ {10, 50, 150} × D ∈ {16, 32} and compare generators. Goal is not SOTA — it is to identify when diffusion is worth the extra complexity over a Gaussian. All training code is my own.

Motivation
3D-LDM uses latent diffusion over DeepSDF codes at full ShapeNet scale; naive Gaussian sampling from the same latent space often fails. I study the minimal-resource regime (small N, small D, Colab) where that tradeoff is unknown.

Validation
Data: ShapeNet subset (~50–200 meshes) or synthetic shapes + stanford_bunny.obj.
Metrics: Chamfer + IoU (reconstruction); Coverage, MMD, 1-NN accuracy (generation); qualitative samples and interpolations.
Mid-point checks: (1) overfit one shape, (2) train DeepSDF on ~10 shapes and extract codes, (3) sample one mesh from Gaussian and one from diffusion.
Final: full N×D grid + generator comparison + degradation curves, including negative results (e.g., Gaussian matches diffusion at small N).

Related work
DeepSDF (Park et al., 2019) — auto-decoder and latent codes (Stage 1).
3D-LDM (Nam et al., 2022) — latent diffusion over DeepSDF codes at scale; does not study small N/D or Gaussian vs diffusion explicitly.
DreamFusion / DDPM (Poole et al., 2022; Ho et al., 2020) — diffusion training (HW3 connection).
