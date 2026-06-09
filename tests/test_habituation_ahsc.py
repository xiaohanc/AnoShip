import anoship.app  # noqa: F401  triggers registration
from anoship.core.registry import DETECTORS
from anoship.detectors._ahsc.ahsc import AHSC
from anoship.signals.synthetic import SyntheticStream


def test_habituation_uses_ahsc_after_fit():
    det = DETECTORS.create("habituation")
    det.fit(SyntheticStream(n_channels=4, seed=7).baseline(600))
    assert isinstance(det._ahsc, AHSC)
    assert len(det._ahsc.space.microclusters) >= 1


def test_habituation_flags_regression_more_than_clean():
    stream = SyntheticStream(n_channels=4, seed=7)
    det = DETECTORS.create("habituation")
    det.fit(stream.baseline(600))

    clean = det.is_anomaly(stream.baseline(120, offset=42))
    reg, _ = stream.inject_regression(
        stream.baseline(120, offset=9), start=30, shift=1.5, channels=[0, 1]
    )
    flagged = det.is_anomaly(reg)

    assert flagged.anomaly_rate > clean.anomaly_rate
    assert flagged.is_anomaly
