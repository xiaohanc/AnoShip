"""Signal channels: model outputs, input features, and downstream indicators.

The framework's central premise is to treat deployment safety as anomaly
detection over evolving streams of three signal families:

* **outputs**     -- the model's predictions / scores,
* **features**    -- the input feature distribution feeding the model,
* **indicators**  -- downstream business / system health indicators.

This module gives those families names and lets the rest of the framework
select sub-streams by role, which is also what enables channel-level root-cause
attribution downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ["ChannelKind", "SignalSchema"]


class ChannelKind:
    OUTPUTS = "outputs"
    FEATURES = "features"
    INDICATORS = "indicators"
    ALL = (OUTPUTS, FEATURES, INDICATORS)


@dataclass
class SignalSchema:
    """Maps named/typed channels onto column indices of a ``(T, C)`` stream."""

    names: List[str]
    roles: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._index = {name: i for i, name in enumerate(self.names)}
        for name, role in self.roles.items():
            if name not in self._index:
                raise ValueError(f"unknown channel in roles: {name!r}")
            if role not in ChannelKind.ALL:
                raise ValueError(f"invalid channel role: {role!r}")

    @classmethod
    def generic(cls, n_channels: int) -> "SignalSchema":
        """A default schema naming channels ch0..chN with no role split."""
        return cls(names=[f"ch{i}" for i in range(n_channels)])

    @property
    def n_channels(self) -> int:
        return len(self.names)

    def index_of(self, name: str) -> int:
        return self._index[name]

    def columns_for(self, role: str) -> List[int]:
        """Column indices whose role matches ``role`` (empty if none set)."""
        return [self._index[n] for n, r in self.roles.items() if r == role]

    def select(self, X: np.ndarray, role: str) -> np.ndarray:
        """Return the sub-stream for ``role``; full stream if no roles set."""
        cols = self.columns_for(role)
        X = np.asarray(X, dtype=float)
        if not cols:
            return X
        return X[:, cols]

    def channel_names(self, cols: Optional[Sequence[int]] = None) -> List[str]:
        if cols is None:
            return list(self.names)
        return [self.names[c] for c in cols]
