"""Metrics collection for deployment runs.

Captures the operational numbers that characterize a deployment-safety system:
how many stages were promoted, whether a rollback fired, detection latency, and
mitigation (time-to-recover) -- reported in abstract "steps" here so the
methodology is reproducible without any proprietary infrastructure.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = ["MetricsCollector", "RunSummary"]


class MetricsCollector:
    def __init__(self) -> None:
        self._counters: Dict[str, float] = defaultdict(float)
        self._values: Dict[str, float] = {}
        self._series: Dict[str, List[float]] = defaultdict(list)

    def incr(self, name: str, amount: float = 1.0) -> None:
        self._counters[name] += amount

    def set(self, name: str, value: float) -> None:
        self._values[name] = float(value)

    def observe(self, name: str, value: float) -> None:
        self._series[name].append(float(value))

    def counter(self, name: str) -> float:
        return self._counters.get(name, 0.0)

    def value(self, name: str) -> Optional[float]:
        return self._values.get(name)

    def snapshot(self) -> Dict[str, object]:
        return {
            "counters": dict(self._counters),
            "values": dict(self._values),
            "series": {k: list(v) for k, v in self._series.items()},
        }


@dataclass
class RunSummary:
    """High-level outcome of a single deployment run."""

    deployment_id: str
    final_state: str
    stages_total: int
    stages_promoted: int
    rolled_back: bool
    detection_delay: Optional[float] = None
    mitigation_time: Optional[float] = None
    root_cause: Optional[str] = None
    extra: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "deployment_id": self.deployment_id,
            "final_state": self.final_state,
            "stages_total": self.stages_total,
            "stages_promoted": self.stages_promoted,
            "rolled_back": self.rolled_back,
            "detection_delay": self.detection_delay,
            "mitigation_time": self.mitigation_time,
            "root_cause": self.root_cause,
            **self.extra,
        }
