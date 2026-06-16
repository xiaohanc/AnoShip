import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")

import anoship.contrib.scid  # noqa: F401  registers the "scid" detector
from anoship.core.errors import NotFittedError
from anoship.core.registry import DETECTORS, THRESHOLDS
from anoship.signals.synthetic import SyntheticStream
from scid_ad.config import SCIDConfig


@pytest.fixture
def stream():
    return SyntheticStream(n_channels=3, seed=7)


def test_scid_is_registered():
    assert "scid" in DETECTORS
    assert "pot" in THRESHOLDS


def test_scid_requires_fit():
    det = DETECTORS.create("scid")
    with pytest.raises(NotFittedError):
        det.score(np.zeros((10, 3)))


def test_scid_uses_quick_config_by_default():
    det = DETECTORS.create("scid")
    assert det.config.window_size == SCIDConfig.quick().window_size


def test_scid_end_to_end(stream):
    det = DETECTORS.create("scid")
    det.fit(stream.baseline(500))

    clean = stream.baseline(120, offset=42)
    scores = det.score(clean)
    assert scores.shape == (clean.shape[0],)

    clean_res = det.is_anomaly(clean)
    assert clean_res.anomaly_rate <= 0.15

    reg, _ = stream.inject_regression(
        stream.baseline(120, offset=9), start=30, shift=2.0, channels=[0, 1]
    )
    reg_res = det.is_anomaly(reg)
    assert reg_res.is_anomaly
    assert reg_res.anomaly_rate > clean_res.anomaly_rate

    # attribution is a normalized distribution over channels
    assert reg_res.attribution is not None
    assert pytest.approx(sum(reg_res.attribution.values()), abs=1e-6) == 1.0


def test_scid_with_pot_threshold(stream):
    from scid_ad.pot import POTThreshold

    det = DETECTORS.create("scid", threshold=POTThreshold())
    det.fit(stream.baseline(500))
    res = det.is_anomaly(stream.baseline(120, offset=3))
    assert np.isfinite(res.threshold)
