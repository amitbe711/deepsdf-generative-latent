"""Metric sanity checks: Chamfer, coverage, MMD, 1-NN accuracy."""

import numpy as np

from src.metrics.generation import (
    chamfer_matrix,
    coverage,
    minimum_matching_distance,
    one_nn_accuracy,
)
from src.metrics.reconstruction import chamfer_distance


def test_chamfer_identical_is_zero():
    pc = np.random.default_rng(0).normal(size=(500, 3)).astype(np.float32)
    assert chamfer_distance(pc, pc) < 1e-8


def test_chamfer_larger_for_shifted():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 3)).astype(np.float32)
    b = a + 1.0
    assert chamfer_distance(a, b) > chamfer_distance(a, a)


def test_generation_metrics_ranges():
    rng = np.random.default_rng(0)
    ref = [rng.normal(size=(200, 3)).astype(np.float32) for _ in range(8)]
    gen = [rng.normal(size=(200, 3)).astype(np.float32) for _ in range(8)]
    mat = chamfer_matrix(gen, ref)
    assert mat.shape == (8, 8)

    cov = coverage(mat)
    mmd = minimum_matching_distance(mat)
    acc = one_nn_accuracy(gen, ref)
    assert 0.0 <= cov <= 1.0
    assert mmd >= 0.0
    assert 0.0 <= acc <= 1.0


def test_one_nn_identical_distributions_near_half():
    # Two samples from the *same* distribution -> 1-NN accuracy close to 0.5.
    rng = np.random.default_rng(1)
    accs = []
    for _ in range(5):
        ref = [rng.normal(size=(300, 3)).astype(np.float32) for _ in range(12)]
        gen = [rng.normal(size=(300, 3)).astype(np.float32) for _ in range(12)]
        accs.append(one_nn_accuracy(gen, ref))
    assert 0.25 <= float(np.mean(accs)) <= 0.75
