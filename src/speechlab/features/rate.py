"""Speech-rate features.

Session 1 reports a GLOBAL rate (words per minute of voiced time). LOCAL per-word
rate needs forced alignment and is deferred to Session 2/4 (word timestamps).
"""
from __future__ import annotations


def voiced_duration_s(silences, total_s: float) -> float:
    """Total non-silent time = clip length minus summed silence."""
    sil = sum(e - s for s, e in silences)
    return max(0.0, total_s - sil)


def speech_rate_wpm(n_words: int, voiced_s: float) -> float:
    """Words per minute over voiced time. 0 if no voiced audio."""
    if voiced_s <= 0:
        return 0.0
    return n_words / (voiced_s / 60.0)
