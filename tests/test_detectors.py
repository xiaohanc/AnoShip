import anoship.app  # noqa: F401  triggers registration
import numpy as np
import pytest
from anoship.core.errors import NotFittedError
from anoship.core.registry import DETECTORS
from anoship.signals.synthetic import SyntheticStream

DETECTOR_NAMES = ["ewma", "habituation", "causal", "diffusion", "spatiotemporal"]


@pytest.fixture
def stream():
    return SyntheticStream(n_channels=4, seed=7)


@pytest.mark.parametrize("name", DETECTOR_NAMES)
def test_score_requires_fit(name):
    det = DETECTORS.create(name)
    with pytest.raises(NotFittedError):
        det.score(np.zeros((10, 4)))


@pytest.mark.parametrize("name", DETECTOR_NAMES)
def test_clean_window_low_rate(name, stream):
    det = DETECTORS.create(name)
    det.fit(stream.baseline(600))
    clean = stream.baseline(120, offset=42)
    result = det.is_anomaly(clean)
    assert 0.0 <= result.anomaly_rate <= 0.15
    assert result.scores.shape[0] == clean.shape[0]


@pytest.mark.parametrize("name", DETECTOR_NAMES)
def test_attribution_normalized(name, stream):
    det = DETECTORS.create(name)
    det.fit(stream.baseline(600))
    reg, _ = stream.inject_regression(
        stream.baseline(120, offset=9), start=30, shift=1.5, channels=[0, 1]
    )
    result = det.is_anomaly(reg)
    assert result.attribution is not None
    assert pytest.approx(sum(result.attribution.values()), abs=1e-6) == 1.0


def test_causal_attributes_to_perturbed_channel(stream):
    det = DETECTORS.create("causal")
    det.fit(stream.baseline(600))
    reg, _ = stream.inject_regression(
        stream.baseline(120, offset=3), start=20, shift=1.5, channels=[0]
    )
    rc = det.is_anomaly(reg)
    top = max(rc.attribution, key=rc.attribution.get)
    assert top == "ch0"


def test_diffusion_robust_to_noise(stream):
    det = DETECTORS.create("diffusion")
    det.fit(stream.baseline(600))
    noisy = stream.add_noise(stream.baseline(120, offset=5), scale=2.0)
    result = det.is_anomaly(noisy)
    # Diffusion denoising should keep the noise-only anomaly rate modest.
    assert result.anomaly_rate < 0.12


def test_ensemble_combines(stream):
    from anoship.detectors.ensemble import EnsembleDetector

    members = [DETECTORS.create(n) for n in ["causal", "diffusion"]]
    ens = EnsembleDetector(members)
    ens.fit(stream.baseline(600))
    reg, _ = stream.inject_regression(
        stream.baseline(120, offset=9), start=30, shift=1.5, channels=[0, 1]
    )
    res = ens.is_anomaly(reg)
    assert res.scores.shape[0] == reg.shape[0]
    assert res.detector == "ensemble"
