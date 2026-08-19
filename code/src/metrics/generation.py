"""Generation metrics (Achlioptas et al., 2018): Coverage, MMD, 1-NN accuracy.

Each shape is represented by a surface point cloud and pairwise distances use
the Chamfer distance. These metrics jointly capture fidelity (MMD), diversity
(Coverage), and distribution match (1-NN accuracy, ideal = 50%).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _prepare(clouds: list[np.ndarray]) -> list[tuple[np.ndarray, cKDTree]]:
    """Pair each cloud with its KD-tree, built once.

    These metrics are quadratic in the number of shapes, so building a tree per
    *pair* -- the obvious implementation -- dominates the cost: 1-NN accuracy
    over G generated and R reference clouds would build 2(G+R)^2 trees where
    G+R suffice. At G=R=200 that is the difference between minutes and hours.
    """
    prepared = []
    for cloud in clouds:
        points = np.asarray(cloud, dtype=np.float32)
        prepared.append((points, cKDTree(points)))
    return prepared


def _chamfer_pair(
    a: np.ndarray, tree_a: cKDTree, b: np.ndarray, tree_b: cKDTree
) -> float:
    """Symmetric Chamfer distance, reusing already-built trees.

    Matches ``reconstruction.chamfer_distance`` exactly; only the trees are hoisted.
    """
    dist_a, _ = tree_b.query(a, workers=-1)
    dist_b, _ = tree_a.query(b, workers=-1)
    return float(np.mean(dist_a**2) + np.mean(dist_b**2))


def _pairwise_chamfer(
    set_a: list[np.ndarray], set_b: list[np.ndarray]
) -> np.ndarray:
    prep_a = _prepare(set_a)
    prep_b = _prepare(set_b)
    matrix = np.zeros((len(prep_a), len(prep_b)), dtype=np.float64)
    for i, (a, tree_a) in enumerate(prep_a):
        for j, (b, tree_b) in enumerate(prep_b):
            matrix[i, j] = _chamfer_pair(a, tree_a, b, tree_b)
    return matrix


def _self_pairwise_chamfer(clouds: list[np.ndarray]) -> np.ndarray:
    """Full square distance matrix of a set against itself, using symmetry."""
    prep = _prepare(clouds)
    num = len(prep)
    matrix = np.zeros((num, num), dtype=np.float64)
    for i in range(num):
        a, tree_a = prep[i]
        for j in range(i + 1, num):
            b, tree_b = prep[j]
            matrix[i, j] = matrix[j, i] = _chamfer_pair(a, tree_a, b, tree_b)
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
    generated: list[np.ndarray],
    reference: list[np.ndarray],
    gen_ref_matrix: np.ndarray | None = None,
) -> float:
    """Leave-one-out 1-NN classifier accuracy over generated (1) vs reference (0).

    A perfect generator yields 0.5 (indistinguishable); values near 0 or 1 mean
    the two distributions are easy to tell apart.

    Pass ``gen_ref_matrix`` from :func:`chamfer_matrix` to skip recomputing the
    cross block, which is a third of the pairs at G=R.
    """
    num_gen, num_ref = len(generated), len(reference)
    labels = np.array([1] * num_gen + [0] * num_ref)
    num = num_gen + num_ref
    if num < 2:
        return float("nan")

    if gen_ref_matrix is None:
        dist = _self_pairwise_chamfer(list(generated) + list(reference))
    else:
        dist = np.empty((num, num), dtype=np.float64)
        dist[:num_gen, :num_gen] = _self_pairwise_chamfer(generated)
        dist[num_gen:, num_gen:] = _self_pairwise_chamfer(reference)
        dist[:num_gen, num_gen:] = gen_ref_matrix
        dist[num_gen:, :num_gen] = gen_ref_matrix.T

    np.fill_diagonal(dist, np.inf)
    nn_idx = dist.argmin(axis=1)
    correct = labels[nn_idx] == labels
    return float(correct.mean())
