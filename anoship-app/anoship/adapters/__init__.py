"""Adapters for plugging external models into the anoship detector interface.

These modules import optional heavyweight dependencies (torch, scikit-learn)
lazily, so importing :mod:`anoship` itself never requires them. Import the
specific adapter you need:

    from anoship.adapters.sklearn import SklearnDetector
    from anoship.adapters.torch_mstdf import TorchReconstructionDetector
"""

from __future__ import annotations

__all__ = ["SklearnDetector", "TorchReconstructionDetector"]


def __getattr__(name: str):  # pragma: no cover - thin lazy import shim
    if name == "SklearnDetector":
        from .sklearn import SklearnDetector

        return SklearnDetector
    if name == "TorchReconstructionDetector":
        from .torch_mstdf import TorchReconstructionDetector

        return TorchReconstructionDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
