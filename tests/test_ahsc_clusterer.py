import numpy as np
import pytest
from anoship.detectors._ahsc.ahsc import AHSC


def _blobs(rng, centers, n_per, scale=0.3):
    pts = [c + rng.normal(scale=scale, size=(n_per, len(c))) for c in centers]
    out = np.vstack(pts)
    rng.shuffle(out)
    return out


def test_outlier_scores_shape_and_requires_fit():
    ahsc = AHSC(seed=0)
    with pytest.raises(RuntimeError):
        ahsc.outlier_scores(np.zeros((5, 4)))
    rng = np.random.default_rng(0)
    X = _blobs(rng, [np.zeros(4), np.full(4, 5.0)], 200)
    ahsc.fit(X)
    s = ahsc.outlier_scores(X[:30])
    assert s.shape == (30,)
    assert np.all(np.isfinite(s))


def test_anomalies_score_higher_than_clean():
    rng = np.random.default_rng(1)
    centers = [
        np.zeros(4),
        np.array([6.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 6.0, 0.0, 0.0]),
    ]
    baseline = _blobs(rng, centers, 250)
    ahsc = AHSC(expansion=20, wta_frac=0.1, n_init=6, seed=1).fit(baseline)

    clean = _blobs(rng, centers, 60)
    # points nowhere near any learned blob
    anomalies = np.full(4, 20.0) + rng.normal(scale=0.3, size=(60, 4))

    clean_scores = ahsc.outlier_scores(clean)
    anomaly_scores = ahsc.outlier_scores(anomalies)
    assert anomaly_scores.mean() > clean_scores.mean()
    # the separation is clear, not marginal
    assert anomaly_scores.mean() > 2.0 * clean_scores.mean()


def test_fit_builds_microclusters():
    rng = np.random.default_rng(2)
    X = _blobs(rng, [np.zeros(3), np.full(3, 8.0)], 150)
    ahsc = AHSC(seed=2).fit(X)
    assert len(ahsc.space.microclusters) >= 2
    assert len(ahsc.space.macro_centers()) >= 1


def test_scoring_is_deterministic_for_fixed_seed():
    rng = np.random.default_rng(3)
    X = _blobs(rng, [np.zeros(4), np.full(4, 5.0)], 200)
    a = AHSC(seed=5).fit(X).outlier_scores(X[:40])
    b = AHSC(seed=5).fit(X).outlier_scores(X[:40])
    assert np.allclose(a, b)
