"""Stage-2 generator: a Gaussian Mixture Model over the frozen latent codes.

This is the *intermediate* prior, sitting between the single Gaussian (unimodal,
linear) and the DDPM (fully expressive) on the minimal-to-expressive spectrum.
A mixture of ``K`` full-covariance Gaussians is fit by Expectation-Maximization
and sampled by first drawing a component, then a Gaussian within it.

Implemented from scratch (NumPy EM) to keep the own-vs-borrowed code accounting
consistent with the rest of ``src/`` (see ``src/thirdparty/README.md``).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def _log_gaussian(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Log-density of N(mean, cov) evaluated at rows of ``x`` -> [N]."""
    dim = x.shape[1]
    diff = x - mean
    # cov is kept positive-definite by the ridge added during fitting.
    inv = np.linalg.inv(cov)
    _, logdet = np.linalg.slogdet(cov)
    maha = np.einsum("ni,ij,nj->n", diff, inv, diff)
    return -0.5 * (dim * np.log(2.0 * np.pi) + logdet + maha)


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    return out


class GMMPrior:
    """Full-covariance Gaussian mixture fit by EM.

    The effective number of components is capped so that each component has, on
    average, at least ``min_points_per_component`` codes; with very small ``N``
    this collapses to a single Gaussian, which is the sensible behaviour.
    """

    def __init__(
        self,
        num_components: int = 5,
        reg: float = 1e-5,
        max_iters: int = 200,
        tol: float = 1e-4,
        min_points_per_component: int = 5,
        seed: int = 0,
    ) -> None:
        self.num_components = num_components
        self.reg = reg
        self.max_iters = max_iters
        self.tol = tol
        self.min_points_per_component = min_points_per_component
        self.seed = seed
        self.weights: np.ndarray | None = None  # (K,)
        self.means: np.ndarray | None = None     # (K, D)
        self.covs: np.ndarray | None = None       # (K, D, D)
        self.chols: np.ndarray | None = None       # (K, D, D)

    def _effective_k(self, num_points: int) -> int:
        by_data = num_points // max(self.min_points_per_component, 1)
        return max(1, min(self.num_components, by_data))

    def _kmeans_init(
        self, x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 10
    ) -> np.ndarray:
        """A few Lloyd iterations to get sensible cluster assignments for EM init.

        A single broad (global-covariance) init makes every component explain
        every point equally and collapses EM to one mode; k-means separates the
        modes first so the mixture actually fits distinct clusters.
        """
        n = x.shape[0]
        centers = x[rng.choice(n, size=k, replace=False)].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(iters):
            dists = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
            labels = dists.argmin(axis=1)
            for j in range(k):
                members = x[labels == j]
                if len(members) > 0:
                    centers[j] = members.mean(axis=0)
        return labels

    def fit(self, codes: Tensor) -> "GMMPrior":
        x = codes.detach().cpu().numpy().astype(np.float64)
        n, dim = x.shape
        k = self._effective_k(n)
        rng = np.random.default_rng(self.seed)

        eye = np.eye(dim)
        if k == 1:
            means = x.mean(axis=0, keepdims=True)
            cov = (np.atleast_2d(np.cov(x.T)) if n > 1 else eye) + self.reg * eye
            covs = cov[None, :, :]
            weights = np.array([1.0])
        else:
            # k-means init: separate the modes before running EM.
            labels = self._kmeans_init(x, k, rng)
            means = np.empty((k, dim))
            covs = np.empty((k, dim, dim))
            weights = np.empty(k)
            global_cov = np.atleast_2d(np.cov(x.T)) + self.reg * eye
            for j in range(k):
                members = x[labels == j]
                weights[j] = max(len(members), 1) / n
                if len(members) >= 2:
                    means[j] = members.mean(axis=0)
                    covs[j] = np.atleast_2d(np.cov(members.T)) + self.reg * eye
                else:
                    means[j] = members.mean(axis=0) if len(members) else x[j]
                    covs[j] = global_cov.copy()
            weights /= weights.sum()

        prev_ll = -np.inf
        for _ in range(self.max_iters):
            # E-step: log responsibilities.
            log_resp = np.empty((n, k))
            for j in range(k):
                log_resp[:, j] = np.log(weights[j] + 1e-12) + _log_gaussian(
                    x, means[j], covs[j]
                )
            log_norm = _logsumexp(log_resp, axis=1)
            log_resp -= log_norm
            resp = np.exp(log_resp)
            ll = float(np.sum(log_norm))

            # M-step.
            nk = resp.sum(axis=0) + 1e-12
            weights = nk / n
            for j in range(k):
                means[j] = (resp[:, j : j + 1] * x).sum(axis=0) / nk[j]
                diff = x - means[j]
                covs[j] = (resp[:, j : j + 1] * diff).T @ diff / nk[j]
                covs[j] += self.reg * np.eye(dim)

            if np.abs(ll - prev_ll) < self.tol * max(np.abs(prev_ll), 1.0):
                break
            prev_ll = ll

        self.weights = weights
        self.means = means
        self.covs = covs
        self.chols = np.stack([np.linalg.cholesky(c) for c in covs])
        return self

    @torch.no_grad()
    def sample(self, num_samples: int, generator: torch.Generator | None = None) -> Tensor:
        if self.means is None:
            raise RuntimeError("GMMPrior must be fit before sampling.")
        # Own RNG for reproducibility; optional torch generator only seeds it.
        seed = self.seed + 1
        if generator is not None:
            seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
        rng = np.random.default_rng(seed)

        dim = self.means.shape[1]
        comps = rng.choice(len(self.weights), size=num_samples, p=self.weights)
        out = np.empty((num_samples, dim), dtype=np.float64)
        for i, j in enumerate(comps):
            eps = rng.standard_normal(dim)
            out[i] = self.means[j] + self.chols[j] @ eps
        return torch.from_numpy(out).float()
