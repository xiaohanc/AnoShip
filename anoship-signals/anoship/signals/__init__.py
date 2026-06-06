"""Signal layer: windowing, typed channels, normalization, synthetic streams."""

from __future__ import annotations

from .channels import ChannelKind, SignalSchema
from .normalize import RobustNormalizer, ZNormalizer
from .synthetic import SyntheticStream
from .window import SlidingWindow, StreamBuffer

__all__ = [
    "ChannelKind",
    "SignalSchema",
    "RobustNormalizer",
    "ZNormalizer",
    "SyntheticStream",
    "SlidingWindow",
    "StreamBuffer",
]
