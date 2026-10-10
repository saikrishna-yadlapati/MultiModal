"""Self-check for the Session 1 slice logic — pure numpy, no SAPI, fast.

Builds a synthetic voiced tone, injects a known 1.5s pause, and asserts the blind
pause detector localizes it. Fails loudly if silence detection, injection, or interval
scoring breaks. Run: python -m pytest tests/test_slice.py  (or: python tests/test_slice.py)
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speechlab.dataset.inject import inject_pause
from speechlab.detection.pause_flaw import detect_long_pauses
from speechlab.evaluation.intervals import interval_iou, start_end_error
from speechlab.features.pauses import detect_silences


def _tone(sr=16000, dur_s=4.0, hz=220.0):
    t = np.arange(int(sr * dur_s)) / sr
    return 0.3 * np.sin(2 * np.pi * hz * t), sr


def test_blind_pause_localization():
    y, sr = _tone()
    cand, gt = inject_pause(y, sr, at_time_s=1.0, pause_s=1.5)

    silences = detect_silences(cand, sr, frame_s=0.025, hop_s=0.010,
                               silence_rel_db=-35.0, min_pause_s=0.15)
    longs = detect_long_pauses(silences, long_pause_s=0.8)

    assert longs, "detector found no long pause"
    pred = (longs[0][0], longs[0][1])
    iou = interval_iou(pred, gt)
    se, ee = start_end_error(pred, gt)
    assert iou > 0.8, f"IoU too low: {iou:.3f}"
    assert se < 0.05 and ee < 0.05, f"edge error too big: {se:.3f}/{ee:.3f}"


def test_interval_iou_basics():
    assert interval_iou((0, 1), (0, 1)) == 1.0
    assert interval_iou((0, 1), (2, 3)) == 0.0
    assert abs(interval_iou((0, 2), (1, 3)) - 1 / 3) < 1e-9


def test_clean_tone_has_no_long_pause():
    y, sr = _tone()
    silences = detect_silences(y, sr, 0.025, 0.010, -35.0, 0.15)
    assert not detect_long_pauses(silences, 0.8), "false positive on clean tone"


if __name__ == "__main__":
    test_blind_pause_localization()
    test_interval_iou_basics()
    test_clean_tone_has_no_long_pause()
    print("all slice self-checks passed")
