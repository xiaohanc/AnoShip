"""Scenario library: reproducible end-to-end deployment situations.

Each scenario bundles an anomaly-free baseline (to fit the detector) with a
per-stage signal source representing a candidate snapshot being rolled out. They
exercise the full pipeline without any external data:

* ``healthy``     -- a good candidate; should roll out to completion.
* ``regression``  -- a subtle, late-appearing relational regression; should be
                     caught and rolled back.
* ``drift``       -- a gradual distributional drift that worsens with exposure.
* ``spike``       -- a sharp transient fault on one channel.
* ``noisy``       -- heavy observation noise but no true fault; should NOT roll
                     back (false-alarm robustness).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..core.registry import register_scenario
from ..pipeline.pipeline import SignalSource, StaticSource
from ..signals.synthetic import SyntheticStream

__all__ = [
    "Scenario",
    "HealthyScenario",
    "RegressionScenario",
    "DriftScenario",
    "SpikeScenario",
    "NoisyScenario",
    "build_scenario",
]


@dataclass
class Scenario:
    name: str
    baseline: np.ndarray
    source: SignalSource
    description: str
    anomalous_from: Optional[int] = None


class _BaseScenario:
    name = "base"
    description = ""

    def __init__(
        self,
        n_channels: int = 4,
        seed: int = 0,
        n_stages: int = 5,
        window: int = 120,
        train: int = 600,
    ) -> None:
        self.n_channels = n_channels
        self.seed = seed
        self.n_stages = n_stages
        self.window = window
        self.train = train

    def _stream(self) -> SyntheticStream:
        return SyntheticStream(n_channels=self.n_channels, seed=self.seed)

    def _stage_windows(self, ss: SyntheticStream) -> List[np.ndarray]:
        raise NotImplementedError

    def build(self) -> Scenario:
        ss = self._stream()
        baseline = ss.baseline(self.train)
        windows = self._stage_windows(ss)
        return Scenario(
            name=self.name,
            baseline=baseline,
            source=StaticSource(windows),
            description=self.description,
            anomalous_from=getattr(self, "anomalous_from", None),
        )


@register_scenario("healthy")
class HealthyScenario(_BaseScenario):
    name = "healthy"
    description = "A healthy candidate that should fully roll out."
    anomalous_from = None

    def _stage_windows(self, ss):
        return [ss.baseline(self.window, offset=100 + i) for i in range(self.n_stages)]


@register_scenario("regression")
class RegressionScenario(_BaseScenario):
    name = "regression"
    description = "A subtle relational regression appearing at higher exposure."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anomalous_from = max(1, self.n_stages - 2)

    def _stage_windows(self, ss):
        windows = []
        for i in range(self.n_stages):
            w = ss.baseline(self.window, offset=100 + i)
            if i >= self.anomalous_from:
                w, _ = ss.inject_regression(
                    w, start=20, shift=1.2, channels=[0, 1]
                )
            windows.append(w)
        return windows


@register_scenario("drift")
class DriftScenario(_BaseScenario):
    name = "drift"
    description = "A gradual drift that intensifies as exposure grows."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anomalous_from = max(1, self.n_stages // 2)

    def _stage_windows(self, ss):
        windows = []
        for i in range(self.n_stages):
            w = ss.baseline(self.window, offset=100 + i)
            if i >= self.anomalous_from:
                slope = 0.04 * (i - self.anomalous_from + 1)
                w, _ = ss.inject_drift(w, start=15, slope=slope, channels=[0])
            windows.append(w)
        return windows


@register_scenario("spike")
class SpikeScenario(_BaseScenario):
    name = "spike"
    description = "A sharp transient fault on a single channel mid-rollout."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anomalous_from = max(1, self.n_stages - 2)

    def _stage_windows(self, ss):
        windows = []
        for i in range(self.n_stages):
            w = ss.baseline(self.window, offset=100 + i)
            if i >= self.anomalous_from:
                w, _ = ss.inject_spike(
                    w, start=40, length=18, magnitude=4.0, channels=[2]
                )
            windows.append(w)
        return windows


@register_scenario("noisy")
class NoisyScenario(_BaseScenario):
    name = "noisy"
    description = "Heavy observation noise but no true fault (false-alarm test)."
    anomalous_from = None

    def _stage_windows(self, ss):
        return [
            ss.add_noise(ss.baseline(self.window, offset=100 + i), scale=2.0)
            for i in range(self.n_stages)
        ]


def build_scenario(name: str, **kwargs) -> Scenario:
    """Instantiate and build a registered scenario by name."""
    from ..core.registry import SCENARIOS

    return SCENARIOS.create(name, **kwargs).build()
