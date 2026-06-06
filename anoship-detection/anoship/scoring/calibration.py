"""Detection-quality metrics for evaluating and calibrating detectors.

These let an author report precision / recall / F1 and detection latency on
labelled streams -- the same operating-point language used to characterize a
production deployment-safety system -- and pick a threshold accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["DetectionMetrics", "evaluate", "best_threshold"]


@dataclass
class DetectionMetrics:
    precision: float
    recall: float
    f1: float
    threshold: float
    detection_delay: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "threshold": self.threshold,
            "detection_delay": self.detection_delay,
        }


def _prf(pred: np.ndarray, labels: np.ndarray) -> tuple:
    tp = int(np.sum((pred == 1) & (labels == 1)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _detection_delay(pred: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Steps between the first true anomaly and its first detection."""
    true_idx = np.flatnonzero(labels == 1)
    if true_idx.size == 0:
        return None
    first_true = int(true_idx[0])
    detected = np.flatnonzero((pred == 1) & (np.arange(len(pred)) >= first_true))
    if detected.size == 0:
        return None
    return float(detected[0] - first_true)


def evaluate(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> DetectionMetrics:
    """Compute precision / recall / F1 / delay at a fixed threshold."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pred = (scores > threshold).astype(int)
    precision, recall, f1 = _prf(pred, labels)
    return DetectionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        threshold=float(threshold),
        detection_delay=_detection_delay(pred, labels),
    )


def best_threshold(
    scores: np.ndarray, labels: np.ndarray, n_steps: int = 100
) -> DetectionMetrics:
    """Grid-search the threshold that maximizes F1 on a labelled stream."""
    scores = np.asarray(scores, dtype=float)
    lo, hi = float(scores.min()), float(scores.max())
    if hi <= lo:
        return evaluate(scores, labels, hi)
    best = None
    for thr in np.linspace(lo, hi, n_steps):
        m = evaluate(scores, labels, float(thr))
        if best is None or m.f1 > best.f1:
            best = m
    assert best is not None
    return best
