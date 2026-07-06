"""GMM prior sanity: fitting recovers clusters, sampling stays finite/in-shape."""

import numpy as np
import torch

from src.models.gmm_prior import GMMPrior


def test_gmm_fits_and_samples_shape():
    torch.manual_seed(0)
    codes = torch.randn(60, 8) * 0.2 + 1.0
    prior = GMMPrior(num_components=3, seed=0).fit(codes)
    samples = prior.sample(20)
    assert samples.shape == (20, 8)
    assert torch.isfinite(samples).all()


def test_gmm_recovers_two_clusters():
    rng = np.random.default_rng(0)
    a = rng.normal(-3.0, 0.15, size=(80, 4))
    b = rng.normal(+3.0, 0.15, size=(80, 4))
    codes = torch.from_numpy(np.concatenate([a, b], axis=0)).float()

    prior = GMMPrior(num_components=2, seed=0).fit(codes)
    assert prior.means.shape[0] == 2
    # The two learned means should straddle the origin (one near -3, one near +3).
    centers = np.sort(prior.means[:, 0])
    assert centers[0] < -1.0 < 1.0 < centers[1]


def test_gmm_collapses_to_single_component_when_few_points():
    codes = torch.randn(6, 16)
    # min_points_per_component=5 -> 6 // 5 == 1 effective component.
    prior = GMMPrior(num_components=5, min_points_per_component=5, seed=0).fit(codes)
    assert prior.means.shape[0] == 1
    assert torch.isfinite(prior.sample(5)).all()
