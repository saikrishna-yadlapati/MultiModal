"""F0 (pitch) contour via Praat (parselmouth). Praat's autocorrelation pitch is the
reference-grade method and ships as a self-contained wheel (no numba/numpy-pin fight).

f0 array uses 0.0 for unvoiced frames, matching Praat's convention.
"""
from __future__ import annotations

import numpy as np
import parselmouth


def f0_contour(y: np.ndarray, sr: int, fmin: float, fmax: float, hop_s: float):
    """Return (times_s, f0_hz). f0_hz == 0.0 where unvoiced."""
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=hop_s, pitch_floor=fmin, pitch_ceiling=fmax)
    f0 = pitch.selected_array["frequency"]
    return np.asarray(pitch.xs()), np.asarray(f0)


def f0_stats(f0: np.ndarray) -> dict:
    """Mean/std/voiced-fraction over voiced frames only (f0 > 0)."""
    voiced = f0[f0 > 0]
    if voiced.size == 0:
        return {"mean_hz": 0.0, "std_hz": 0.0, "voiced_fraction": 0.0}
    return {
        "mean_hz": float(voiced.mean()),
        "std_hz": float(voiced.std()),
        "voiced_fraction": float(voiced.size / f0.size),
    }
