"""PyTorch adapter: plug a real published model into the anoship interface.

The companion repository ``MSTDF-AD`` (Modeling Spatiotemporal Dependency Fusion
for Non-Stationary Time Series Anomaly Detection, Information Processing &
Management 2026) is a full PyTorch implementation of one of the detectors
reference-implemented in :mod:`anoship.detectors.spatiotemporal`. This adapter
lets that exact model -- or any reconstruction-based ``torch.nn.Module`` -- drive
the same progressive-rollout health gate.

Requires the optional ``torch`` extra.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..core.registry import register_detector
from ..detectors.base import BaseDetector

__all__ = ["TorchReconstructionDetector"]


@register_detector("torch_recon")
class TorchReconstructionDetector(BaseDetector):
    """Scores per-row reconstruction error from a PyTorch model.

    Parameters
    ----------
    model:
        A ``torch.nn.Module`` mapping a ``(T, C)`` window (as a tensor) to a
        reconstruction of the same shape. For sequence models that expect a
        batch/seq dimension, pass a ``forward_fn`` to adapt the call.
    forward_fn:
        Optional ``(model, tensor) -> tensor`` override controlling how the
        model is invoked; defaults to ``model(tensor)``.
    device:
        Torch device string.
    """

    name = "torch_recon"

    def __init__(
        self,
        model,
        forward_fn: Optional[Callable] = None,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        try:
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "PyTorch is required for TorchReconstructionDetector; install "
                "with 'pip install anoship[torch]'"
            ) from exc
        self._torch = __import__("torch")
        self.model = model.to(device)
        self.device = device
        self.forward_fn = forward_fn or (lambda m, x: m(x))

    def _fit(self, X: np.ndarray) -> None:
        # The wrapped model is assumed pretrained; calibration of the threshold
        # over baseline reconstruction error is handled by BaseDetector.fit.
        self.model.eval()

    def _score(self, X: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            recon = self.forward_fn(self.model, tensor)
            recon_np = np.asarray(recon.detach().cpu().numpy(), dtype=float)
        recon_np = recon_np.reshape(X.shape)
        return np.linalg.norm(X - recon_np, axis=1)
