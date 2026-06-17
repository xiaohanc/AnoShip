import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("scid_ad")

from scid_ad.causal import build_causal_graph, ksg_cmi, ksg_mi, normalized_cmi
from scid_ad.config import SCIDConfig


def test_mi_high_for_dependent():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    y = x + 0.05 * rng.normal(size=2000)  # near-deterministic dependence
    mi = ksg_mi(x, y, k=5)
    assert mi > 0.5


def test_mi_near_zero_for_independent():
    rng = np.random.default_rng(1)
    x = rng.normal(size=2000)
    y = rng.normal(size=2000)
    mi = ksg_mi(x, y, k=5)
    assert abs(mi) < 0.05


def test_cmi_drops_when_conditioning_on_cause():
    # x -> y, x -> w. Conditioning y on x should explain away y~w dependence.
    rng = np.random.default_rng(2)
    x = rng.normal(size=2000)
    y = x + 0.1 * rng.normal(size=2000)
    w = x + 0.1 * rng.normal(size=2000)
    mi_yw = ksg_mi(y, w, k=5)
    cmi_yw_given_x = ksg_cmi(y, w, x, k=5)
    assert mi_yw > 0.3
    assert cmi_yw_given_x < mi_yw
    assert cmi_yw_given_x < 0.2


def test_normalized_cmi_in_unit_interval():
    rng = np.random.default_rng(3)
    x = rng.normal(size=1000)
    y = x + 0.1 * rng.normal(size=1000)
    nc = normalized_cmi(x, y, k=5)
    assert 0.0 <= nc < 1.0
    nc_indep = normalized_cmi(rng.normal(size=1000), rng.normal(size=1000), k=5)
    assert 0.0 <= nc_indep < 0.1


def test_constant_signal_guarded():
    rng = np.random.default_rng(4)
    x = rng.normal(size=500)
    const = np.full(500, 3.0)
    mi = ksg_mi(x, const, k=5)
    assert np.isfinite(mi)
    assert mi < 0.1


def test_build_causal_graph_recovers_edge():
    rng = np.random.default_rng(5)
    n = 1500
    x0 = rng.normal(size=n)
    x1 = x0 + 0.1 * rng.normal(size=n)  # strongly coupled to x0
    x2 = rng.normal(size=n)  # independent
    x3 = rng.normal(size=n)  # independent
    X = np.column_stack([x0, x1, x2, x3])

    cfg = SCIDConfig.quick()
    S, adj = build_causal_graph(X, cfg)
    assert S.shape == (4, 4)
    assert adj.shape == (4, 4)
    assert not np.any(np.diag(adj))  # no self edges
    # the x0<->x1 coupling should be among the strongest off-diagonal scores
    assert S[0, 1] > S[2, 3]
    assert S[0, 1] > S[0, 2]


def test_build_causal_graph_single_variable():
    X = np.random.default_rng(6).normal(size=(100, 1))
    S, adj = build_causal_graph(X, SCIDConfig.quick())
    assert S.shape == (1, 1)
    assert not adj.any()
