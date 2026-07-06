"""Stage 2 (generator #3): fit a Gaussian Mixture Model to the frozen codes."""

from __future__ import annotations

from typing import Any

import torch

from ..models.gmm_prior import GMMPrior


def fit_gmm(cfg: Any, codes: torch.Tensor) -> GMMPrior:
    """Fit a GMM prior to ``codes`` [N, D] using the configured hyper-parameters."""
    gmm_cfg = cfg.gmm
    prior = GMMPrior(
        num_components=int(gmm_cfg.get("num_components", 5)),
        reg=float(gmm_cfg.get("reg", 1e-5)),
        max_iters=int(gmm_cfg.get("max_iters", 200)),
        min_points_per_component=int(gmm_cfg.get("min_points_per_component", 5)),
        seed=int(cfg.seed),
    )
    prior.fit(codes)
    return prior
