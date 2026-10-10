"""Interval scoring. Reused by the Session 1 slice and the Session 6 eval harness."""
from __future__ import annotations


def interval_iou(a, b) -> float:
    """Intersection-over-union of two [start, end] intervals (seconds)."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def start_end_error(pred, true):
    """Absolute (start_error_s, end_error_s) between predicted and true interval."""
    return abs(pred[0] - true[0]), abs(pred[1] - true[1])
