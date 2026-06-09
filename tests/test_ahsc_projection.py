import numpy as np
import pytest
from anoship.detectors._ahsc.projection import FlyProjection, winner_take_all


def test_projection_matrix_is_sparse_binary_with_expansion():
    proj = FlyProjection(input_dim=4, expansion=40, density=0.1, seed=0)
    M = proj.matrix
    assert M.shape == (160, 4)  # m = expansion * d
    assert set(np.unique(M)).issubset({0.0, 1.0})
    # sparse: far fewer than half the entries are connected
    assert M.mean() < 0.5
    # every Kenyon row samples at least one projection neuron
    assert np.all(M.sum(axis=1) >= 1)


def test_transform_is_matrix_projection_Y_equals_MX():
    proj = FlyProjection(input_dim=3, expansion=10, density=0.3, seed=1)
    X = np.random.default_rng(2).normal(size=(20, 3))
    Y = proj.transform(X)
    assert Y.shape == (20, 30)
    assert np.allclose(Y, X @ proj.matrix.T)


def test_projection_is_reproducible_with_seed():
    a = FlyProjection(input_dim=5, expansion=8, density=0.2, seed=7).matrix
    b = FlyProjection(input_dim=5, expansion=8, density=0.2, seed=7).matrix
    assert np.array_equal(a, b)


def test_winner_take_all_keeps_top_fraction_and_zeros_rest():
    Y = np.array([[1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 4.0, 7.0, 6.0, 0.0]])
    S = winner_take_all(Y, frac=0.2)  # keep top 2 of 10
    assert (S != 0).sum() == 2
    # the two survivors are the largest activations
    assert S[0, 5] == 9.0 and S[0, 3] == 8.0
    assert S[0, 0] == 0.0


def test_winner_take_all_keeps_at_least_one():
    Y = np.array([[0.1, 0.2, 0.3]])
    S = winner_take_all(Y, frac=0.01)
    assert (S != 0).sum() == 1
    assert S[0, 2] == 0.3


def test_winner_take_all_is_row_wise():
    Y = np.array([[1.0, 9.0, 2.0], [9.0, 1.0, 2.0]])
    S = winner_take_all(Y, frac=0.34)  # top 1 per row
    assert S[0, 1] == 9.0 and S[0, 0] == 0.0
    assert S[1, 0] == 9.0 and S[1, 1] == 0.0
