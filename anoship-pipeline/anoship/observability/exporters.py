"""Observability exporters: where events go.

* ``console`` -- human-readable, color-free timeline for terminals and logs.
* ``json``    -- machine-readable events for dashboards or audit storage.
* ``null``    -- discard (useful in tests / library embedding).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from typing import List, Optional, TextIO

from ..core.interfaces import Exporter
from ..core.registry import register_exporter
from ..core.types import Event

__all__ = ["ConsoleExporter", "JSONExporter", "NullExporter"]


@register_exporter("console")
class ConsoleExporter(Exporter):
    def __init__(self, stream: Optional[TextIO] = None, verbose: bool = True) -> None:
        self._stream = stream or sys.stdout
        self.verbose = verbose

    def export(self, event: Event) -> None:
        ts = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
        stage = f"[{event.stage}] " if event.stage else ""
        line = f"{ts}  {event.kind.value:22s} {stage}{event.message}"
        if self.verbose and event.payload:
            extras = " ".join(f"{k}={v}" for k, v in event.payload.items())
            line += f"  ({extras})"
        print(line, file=self._stream)


@register_exporter("json")
class JSONExporter(Exporter):
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self.records: List[dict] = []

    def export(self, event: Event) -> None:
        self.records.append(event.to_dict())

    def flush(self) -> None:
        if self.path:
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(self.records, handle, indent=2)

    def as_json(self) -> str:
        return json.dumps(self.records, indent=2)


@register_exporter("null")
class NullExporter(Exporter):
    def export(self, event: Event) -> None:
        return None
