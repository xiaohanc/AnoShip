"""Progressive rollout strategies.

Each strategy defines *how* a candidate snapshot is exposed over time -- the
"progressive rollout through staged regional or traffic exposure" component of
the methodology. New strategies plug in by subclassing ``RolloutStrategy`` and
registering a name.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..core.interfaces import RolloutStrategy
from ..core.registry import register_rollout
from ..core.types import RolloutStage

__all__ = [
    "CanaryRollout",
    "PercentageRollout",
    "RegionalRollout",
    "BlueGreenRollout",
]


@register_rollout("percentage")
class PercentageRollout(RolloutStrategy):
    """Expose to a growing percentage of traffic."""

    def __init__(self, steps: Sequence[float] = (0.1, 0.25, 0.5, 1.0)) -> None:
        self.steps = list(steps)

    def stages(self) -> List[RolloutStage]:
        return [
            RolloutStage(name=f"{int(p * 100)}%", exposure=float(p), index=i)
            for i, p in enumerate(self.steps)
        ]


@register_rollout("canary")
class CanaryRollout(RolloutStrategy):
    """A tiny canary stage first, then a percentage ramp."""

    def __init__(
        self,
        canary: float = 0.01,
        steps: Sequence[float] = (0.05, 0.25, 0.5, 1.0),
    ) -> None:
        self.canary = float(canary)
        self.steps = list(steps)

    def stages(self) -> List[RolloutStage]:
        stages = [RolloutStage(name="canary", exposure=self.canary, index=0)]
        for i, p in enumerate(self.steps, start=1):
            stages.append(
                RolloutStage(name=f"{int(p * 100)}%", exposure=float(p), index=i)
            )
        return stages


@register_rollout("regional")
class RegionalRollout(RolloutStrategy):
    """Roll out region by region, accumulating exposure as regions go live."""

    def __init__(
        self, regions: Optional[Sequence[str]] = None
    ) -> None:
        self.regions = list(
            regions or ["us-east", "us-west", "eu", "apac"]
        )

    def stages(self) -> List[RolloutStage]:
        n = len(self.regions)
        return [
            RolloutStage(
                name=region,
                exposure=float((i + 1) / n),
                index=i,
                region=region,
            )
            for i, region in enumerate(self.regions)
        ]


@register_rollout("blue_green")
class BlueGreenRollout(RolloutStrategy):
    """Shadow the candidate at zero user exposure, then cut over fully."""

    def stages(self) -> List[RolloutStage]:
        return [
            RolloutStage(name="shadow", exposure=0.0, index=0),
            RolloutStage(name="cutover", exposure=1.0, index=1),
        ]
