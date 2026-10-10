"""Controlled flaw injection — Session 1 minimal subset: insert one known pause.

Full seeded, multi-flaw variant generators arrive in Session 3. This is the single
transform the vertical slice needs: it returns the EXACT ground-truth interval of the
inserted silence so blind detection can be scored honestly.
"""
from __future__ import annotations

import numpy as np


def inject_pause(y: np.ndarray, sr: int, at_time_s: float, pause_s: float):
    """Insert `pause_s` of silence at `at_time_s`.

    Returns (new_y, ground_truth_interval). The interval is computed from the
    sample-rounded insertion point, so it is exact for the returned audio.
    """
    idx = min(max(int(round(at_time_s * sr)), 0), len(y))
    n_sil = int(round(pause_s * sr))
    silence = np.zeros(n_sil, dtype=y.dtype)
    new_y = np.concatenate([y[:idx], silence, y[idx:]])
    gt = (idx / sr, (idx + n_sil) / sr)
    return new_y, gt
