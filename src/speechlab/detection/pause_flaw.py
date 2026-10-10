"""Blind long-pause detector.

BLIND means: it sees only the candidate's own silences and a threshold. It is NOT
given the ground-truth injection time. A long pause is any silence >= long_pause_s.

Session 1 uses an absolute threshold on the candidate. Baseline-relative comparison
(a pause the baseline does not have at the aligned position) comes with alignment in
later sessions; the slice already prints baseline pauses for contrast.
"""
from __future__ import annotations


def detect_long_pauses(silences, long_pause_s: float):
    """Return silences (start_s, end_s, duration_s) with duration >= long_pause_s."""
    out = []
    for s, e in silences:
        dur = e - s
        if dur >= long_pause_s:
            out.append((s, e, dur))
    return out
