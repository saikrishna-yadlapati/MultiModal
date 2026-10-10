"""Pause / silence features from short-time RMS energy.

Pure numpy. No heavy audio dependency needed for energy-based silence detection.
A "silence" is a run of frames whose RMS sits >= |silence_rel_db| below the clip's
peak RMS. A "pause" is a silence at least `min_pause_s` long.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-10


def frame_rms(y: np.ndarray, sr: int, frame_s: float, hop_s: float):
    """Return (times_s, rms) for short-time frames. times_s = frame centers."""
    frame = max(1, int(round(frame_s * sr)))
    hop = max(1, int(round(hop_s * sr)))
    if len(y) < frame:
        # One short frame over the whole clip.
        rms = np.array([np.sqrt(np.mean(y ** 2)) if len(y) else 0.0])
        return np.array([len(y) / (2 * sr)]), rms
    n = 1 + (len(y) - frame) // hop
    rms = np.empty(n)
    times = np.empty(n)
    for i in range(n):
        seg = y[i * hop: i * hop + frame]
        rms[i] = np.sqrt(np.mean(seg ** 2))
        times[i] = (i * hop + frame / 2) / sr
    return times, rms


def detect_silences(y: np.ndarray, sr: int, frame_s: float, hop_s: float,
                    silence_rel_db: float, min_pause_s: float):
    """Detect silence intervals. Returns list of (start_s, end_s), each >= min_pause_s.

    Threshold is relative to the clip's own peak RMS, so it is level-independent.
    """
    times, rms = frame_rms(y, sr, frame_s, hop_s)
    ref = rms.max()
    if ref <= _EPS:
        return [(0.0, len(y) / sr)]  # whole clip is silence
    db = 20.0 * np.log10(rms / ref + _EPS)
    silent = db < silence_rel_db

    intervals = []
    i = 0
    half = hop_s / 2.0
    while i < len(silent):
        if silent[i]:
            j = i
            while j + 1 < len(silent) and silent[j + 1]:
                j += 1
            start = max(0.0, times[i] - half)
            end = min(len(y) / sr, times[j] + half)
            if end - start >= min_pause_s:
                intervals.append((start, end))
            i = j + 1
        else:
            i += 1
    return intervals
