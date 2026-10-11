"""Pure self-checks for the Session 6b event-matching + metrics math (no audio/model)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from speechlab.evaluation.metrics import (  # noqa: E402
    localization_stats, match_events, prf1,
)


def test_perfect_fp_fn():
    preds = [("long_pause", (4.0, 6.0)), ("monotone", (1.0, 2.0))]
    gts = [("long_pause", (4.0, 6.0)), ("rate_deviation", (8.0, 9.0))]
    r = match_events(preds, gts, 0.3)
    assert len(r.tp) == 1 and len(r.fp) == 1 and len(r.fn) == 1


def test_class_must_match():
    # Same interval, wrong class -> not a match.
    r = match_events([("monotone", (4.0, 6.0))], [("long_pause", (4.0, 6.0))], 0.3)
    assert len(r.tp) == 0 and len(r.fp) == 1 and len(r.fn) == 1


def test_iou_threshold_respected():
    r = match_events([("long_pause", (4.0, 6.0))], [("long_pause", (5.9, 7.9))], 0.3)
    assert len(r.tp) == 0  # overlap 0.1s / union 3.9s = 0.026 < 0.3


def test_no_double_counting():
    # Two predictions, one true event -> one TP, one FP (not two TPs).
    preds = [("long_pause", (4.0, 6.0)), ("long_pause", (4.1, 6.1))]
    gts = [("long_pause", (4.0, 6.0))]
    r = match_events(preds, gts, 0.3)
    assert len(r.tp) == 1 and len(r.fp) == 1 and len(r.fn) == 0


def test_prf1_values():
    p, r, f = prf1(1, 1, 1)
    assert abs(p - 0.5) < 1e-9 and abs(r - 0.5) < 1e-9 and abs(f - 0.5) < 1e-9
    assert prf1(0, 0, 0) == (0.0, 0.0, 0.0)  # zero-safe


def test_localization_empty_safe():
    assert localization_stats([])["n"] == 0
