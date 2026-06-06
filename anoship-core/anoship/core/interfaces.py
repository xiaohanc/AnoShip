"""Abstract interfaces that define anoship's extension points.

Every pluggable component in the framework implements one of these ABCs. This
is what makes anoship a *framework* rather than a script: detectors, rollout
strategies, gate policies, thresholding strategies, and observability exporters
can all be swapped or extended without touching the orchestration core.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, TYPE_CHECKING

import numpy as np

from .types import AnomalyResult, Event, GateDecision, RolloutStage

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .context import RunContext

__all__ = [
    "Detector",
    "ThresholdStrategy",
    "RolloutStrategy",
    "GatePolicy",
    "Exporter",
]


class Detector(ABC):
    """Pluggable anomaly-detection method.

    A detector learns a notion of "normal" from baseline data and then scores
    new windows of streaming signals. Concrete detectors in
    :mod:`anoship.detectors` implement the published methods underlying this
    framework; any third-party method can participate simply by subclassing
    this interface.
    """

    #: Human-readable, registry-unique detector name.
    name: str = "detector"

    @abstractmethod
    def fit(self, X: np.ndarray) -> "Detector":
        """Learn baseline behavior from normal data of shape ``(T, C)``."""

    @abstractmethod
    def score(self, window: np.ndarray) -> np.ndarray:
        """Return per-row anomaly scores for ``window`` of shape ``(T, C)``."""

    @abstractmethod
    def is_anomaly(self, window: np.ndarray) -> AnomalyResult:
        """Score ``window`` and apply thresholding to produce a decision."""


class ThresholdStrategy(ABC):
    """Maps a stream of anomaly scores to a decision threshold."""

    @abstractmethod
    def fit(self, scores: np.ndarray) -> "ThresholdStrategy":
        """Calibrate the threshold from baseline scores."""

    @abstractmethod
    def threshold(self, scores: np.ndarray) -> float:
        """Return the threshold to apply to ``scores``."""


class RolloutStrategy(ABC):
    """Defines how a candidate snapshot is progressively exposed."""

    @abstractmethod
    def stages(self) -> List[RolloutStage]:
        """Return the ordered list of rollout stages."""


class GatePolicy(ABC):
    """Turns an :class:`AnomalyResult` into a promote/hold/rollback decision."""

    name: str = "policy"

    @abstractmethod
    def evaluate(
        self,
        anomaly_result: AnomalyResult,
        stage: RolloutStage,
        context: "RunContext",
    ) -> GateDecision:
        """Decide what to do at ``stage`` given ``anomaly_result``."""


class Exporter(ABC):
    """Receives observability events as they happen."""

    @abstractmethod
    def export(self, event: Event) -> None:
        """Handle a single event."""

    def export_many(self, events: Sequence[Event]) -> None:
        for event in events:
            self.export(event)

    def flush(self) -> None:  # pragma: no cover - optional hook
        """Flush any buffered state. Override if needed."""
