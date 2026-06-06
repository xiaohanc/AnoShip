"""Promotion controller: advance a healthy stage to the next exposure level."""

from __future__ import annotations

from ..core.context import RunContext
from ..core.types import EventKind, RolloutStage

__all__ = ["PromotionController"]


class PromotionController:
    def promote(self, context: RunContext, stage: RolloutStage) -> None:
        if context.metrics is not None:
            context.metrics.incr("stages_promoted")
        context.emit(
            EventKind.PROMOTED,
            f"stage '{stage.name}' healthy; promoting",
            stage=stage.name,
            exposure=stage.exposure,
        )
