"""Generation metrics (Achlioptas et al., 2018): Coverage, MMD, 1-NN accuracy.

Each shape is represented by a surface point cloud and pairwise distances use
the Chamfer distance. These metrics jointly capture fidelity (MMD), diversity
(Coverage), and distribution match (1-NN accuracy, ideal = 50%).
"""

from __future__ import annotations

import numpy as np

from .reconstruction import chamfer_distance


def _pairwise_chamfer(
    set_a: list[np.ndarray], set_b: list[np.ndarray]
) -> np.ndarray:
    matrix = np.zeros((len(set_a), len(set_b)), dtype=np.float64)
    for i, a in enumerate(set_a):
        for j, b in enumerate(set_b):
            matrix[i, j] = chamfer_distance(a, b)
    return matrix


def chamfer_matrix(
    generated: list[np.ndarray], reference: list[np.ndarray]
) -> np.ndarray:
    """(G, R) matrix of Chamfer distances between generated and reference clouds."""
    return _pairwise_chamfer(generated, reference)


def minimum_matching_distance(gen_ref_matrix: np.ndarray) -> float:
    """MMD-CD: for each reference, distance to its nearest generated sample."""
    if gen_ref_matrix.size == 0:
        return float("nan")
    return float(gen_ref_matrix.min(axis=0).mean())


def coverage(gen_ref_matrix: np.ndarray) -> float:
    """Fraction of references that are the nearest neighbor of some generated sample."""
    if gen_ref_matrix.size == 0:
        return float("nan")
    num_ref = gen_ref_matrix.shape[1]
    matched = np.unique(gen_ref_matrix.argmin(axis=1))
    return float(len(matched)) / float(num_ref)


def one_nn_accuracy(
    generated: list[np.ndarray], reference: list[np.ndarray]
) -> float:
    """Leave-one-out 1-NN classifier accuracy over generated (1) vs reference (0).

    A perfect generator yields 0.5 (indistinguishable); values near 0 or 1 mean
    the two distributions are easy to tell apart.
    """
    combined = list(generated) + list(reference)
    labels = np.array([1] * len(generated) + [0] * len(reference))
    num = len(combined)
    if num < 2:
        return float("nan")

    dist = _pairwise_chamfer(combined, combined)
    np.fill_diagonal(dist, np.inf)
    nn_idx = dist.argmin(axis=1)
    correct = labels[nn_idx] == labels
    return float(correct.mean())
