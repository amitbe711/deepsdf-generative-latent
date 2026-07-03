"""Stage 2 (generator #1): fit a Gaussian to the frozen latent codes."""

from __future__ import annotations

from typing import Any

import torch

from ..models.gaussian_prior import GaussianPrior


def fit_gaussian(cfg: Any, codes: torch.Tensor) -> GaussianPrior:
    """Fit a Gaussian prior to ``codes`` [N, D] using the configured covariance."""
    prior = GaussianPrior(
        covariance=str(cfg.gaussian.covariance),
        reg=float(cfg.gaussian.reg),
    )
    prior.fit(codes)
    return prior
