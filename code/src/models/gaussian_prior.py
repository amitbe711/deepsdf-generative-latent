"""Stage-2 generator #1: a Gaussian fit to the frozen latent codes.

This is the *minimal* prior baseline. We fit a multivariate Gaussian to the set
of DeepSDF codes {z_i} and sample from it. Two variants are supported:
  * ``diagonal``  - per-dimension mean/variance (matches the implicit N(0, I)
    assumption behind DeepSDF's L2 code regularizer).
  * ``full``      - full covariance with a Cholesky factor, capturing linear
    correlations between latent dimensions.
"""

from __future__ import annotations

import torch
from torch import Tensor


class GaussianPrior:
    def __init__(self, covariance: str = "full", reg: float = 1e-5) -> None:
        if covariance not in ("diagonal", "full"):
            raise ValueError(f"Unknown covariance type: {covariance!r}")
        self.covariance = covariance
        self.reg = reg
        self.mean: Tensor | None = None
        self.cov: Tensor | None = None
        self.chol: Tensor | None = None
        self.std: Tensor | None = None

    def fit(self, codes: Tensor) -> "GaussianPrior":
        codes = codes.detach().float()
        dim = codes.shape[1]
        self.mean = codes.mean(dim=0)
        if self.covariance == "diagonal":
            self.std = codes.std(dim=0, unbiased=True).clamp_min(self.reg**0.5)
        else:
            centered = codes - self.mean
            cov = (centered.t() @ centered) / max(codes.shape[0] - 1, 1)
            cov = cov + self.reg * torch.eye(dim, dtype=cov.dtype)
            self.cov = cov
            self.chol = torch.linalg.cholesky(cov)
        return self

    @torch.no_grad()
    def sample(self, num_samples: int, generator: torch.Generator | None = None) -> Tensor:
        if self.mean is None:
            raise RuntimeError("GaussianPrior must be fit before sampling.")
        dim = self.mean.shape[0]
        eps = torch.randn(num_samples, dim, generator=generator)
        if self.covariance == "diagonal":
            return self.mean + eps * self.std
        return self.mean + eps @ self.chol.t()

    def state_dict(self) -> dict[str, object]:
        return {
            "covariance": self.covariance,
            "reg": self.reg,
            "mean": self.mean,
            "cov": self.cov,
            "chol": self.chol,
            "std": self.std,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "GaussianPrior":
        prior = cls(covariance=state["covariance"], reg=state["reg"])
        prior.mean = state["mean"]
        prior.cov = state["cov"]
        prior.chol = state["chol"]
        prior.std = state["std"]
        return prior
