import numpy as np
import pytest

from anoship.scoring.thresholds import (
    AdaptiveThreshold,
    QuantileThreshold,
    SigmaThreshold,
    StaticThreshold,
)
from anoship.scoring.fusion import fuse_scores, robust_zscale
from anoship.scoring.calibration import best_threshold, evaluate


def test_static_threshold():
    t = StaticThreshold(2.5).fit(np.zeros(10))
    assert t.threshold(np.array([1.0])) == 2.5


def test_sigma_threshold():
    scores = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    t = SigmaThreshold(k=2.0).fit(scores)
    assert t.threshold(scores) > scores.mean()


def test_quantile_threshold():
    scores = np.linspace(0, 1, 101)
    t = QuantileThreshold(q=0.9).fit(scores)
    assert 0.85 <= t.threshold(scores) <= 0.95


def test_adaptive_threshold_has_floor():
    base = np.random.default_rng(0).normal(size=500)
    t = AdaptiveThreshold(k=3.0).fit(base)
    # Threshold on quiet input should not drop below the calibrated floor.
    quiet = np.zeros(20)
    assert t.threshold(quiet) >= t._floor - 1e-9


def test_robust_zscale_centers():
    x = np.array([10.0, 10.0, 10.0, 50.0])
    z = robust_zscale(x)
    assert z[-1] > z[0]


def test_fuse_scores_methods():
    a = np.array([0.0, 1.0, 2.0])
    b = np.array([2.0, 1.0, 0.0])
    mean = fuse_scores([a, b], method="mean")
    assert mean.shape == (3,)
    with pytest.raises(ValueError):
        fuse_scores([a, b], method="weighted")  # missing weights
    w = fuse_scores([a, b], method="weighted", weights=[1.0, 0.0])
    assert np.allclose(w, robust_zscale(a))


def test_calibration_metrics():
    scores = np.array([0.1, 0.2, 0.9, 0.95, 0.15])
    labels = np.array([0, 0, 1, 1, 0])
    m = evaluate(scores, labels, threshold=0.5)
    assert m.precision == 1.0 and m.recall == 1.0
    best = best_threshold(scores, labels)
    assert best.f1 == 1.0
