"""Event bus: the backbone of anoship's observability layer.

Every meaningful moment in a deployment (stage entered, gate evaluated, rollback
triggered) is published as a structured :class:`Event`. Exporters subscribe to
the bus, and the full ordered history is retained so a run can be replayed,
audited, or reported on afterwards -- the "standardized observability" component
of the methodology.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..core.interfaces import Exporter
from ..core.types import Event, EventKind

__all__ = ["EventBus"]


class EventBus:
    def __init__(self, exporters: Optional[Sequence[Exporter]] = None) -> None:
        self._exporters: List[Exporter] = list(exporters or [])
        self.history: List[Event] = []

    def subscribe(self, exporter: Exporter) -> None:
        self._exporters.append(exporter)

    def publish(self, event: Event) -> Event:
        self.history.append(event)
        for exporter in self._exporters:
            exporter.export(event)
        return event

    def events_of(self, kind: EventKind) -> List[Event]:
        return [e for e in self.history if e.kind == kind]

    def flush(self) -> None:
        for exporter in self._exporters:
            exporter.flush()
