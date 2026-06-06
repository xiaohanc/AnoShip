"""Snapshot registry: tracks model snapshot versions and the safe baseline.

Automated rollback needs a well-defined "last known good" to restore. This
registry records each candidate snapshot, marks which have passed their health
gates, and always knows the safe snapshot to revert to.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..core.errors import RollbackError
from ..core.types import Snapshot

__all__ = ["SnapshotRegistry"]


class SnapshotRegistry:
    def __init__(self) -> None:
        self._snapshots: Dict[str, Snapshot] = {}
        self._order: List[str] = []
        self._safe_id: Optional[str] = None
        self._counter = 0

    def register(
        self,
        snapshot_id: str,
        meta: Optional[dict] = None,
        safe: bool = False,
    ) -> Snapshot:
        if snapshot_id in self._snapshots:
            raise RollbackError(f"snapshot {snapshot_id!r} already registered")
        snap = Snapshot(
            snapshot_id=snapshot_id,
            version=self._counter,
            is_safe=safe,
            meta=meta or {},
        )
        self._snapshots[snapshot_id] = snap
        self._order.append(snapshot_id)
        self._counter += 1
        if safe:
            self._safe_id = snapshot_id
        return snap

    def mark_safe(self, snapshot_id: str) -> None:
        if snapshot_id not in self._snapshots:
            raise RollbackError(f"unknown snapshot {snapshot_id!r}")
        self._snapshots[snapshot_id].is_safe = True
        self._safe_id = snapshot_id

    def get(self, snapshot_id: str) -> Snapshot:
        return self._snapshots[snapshot_id]

    @property
    def safe_snapshot(self) -> Optional[Snapshot]:
        if self._safe_id is None:
            return None
        return self._snapshots[self._safe_id]

    def rollback_target(self) -> Snapshot:
        """Return the snapshot a rollback should restore, or raise."""
        if self.safe_snapshot is None:
            raise RollbackError("no safe snapshot available to roll back to")
        return self.safe_snapshot

    def all(self) -> List[Snapshot]:
        return [self._snapshots[s] for s in self._order]
