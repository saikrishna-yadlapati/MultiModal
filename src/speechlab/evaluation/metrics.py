"""Event-level metrics for the held-out evaluation (Session 6b).

Pure functions — no audio, no model, no manifest I/O — so the scoring math is unit
testable on its own. `scripts/run_eval.py` feeds these the blind detector output and
the revealed ground truth.

Vocabulary (all "event-level": one flaw = one event, located in time):
  - A PREDICTION is (flaw_type, [start, end]) emitted by a detector.
  - A GROUND-TRUTH event is (flaw_type, [start, end]) from the manifest.
  - A prediction MATCHES a ground-truth event when they are the SAME flaw_type AND
    their intervals overlap by IoU >= iou_thr (intersection-over-union; see intervals.py).
  - True positive  (TP): a matched prediction.
  - False positive (FP): a prediction that matched nothing (a false alarm).
  - False negative (FN): a ground-truth event no prediction matched (a miss).

Precision = TP / (TP + FP)   "of my alarms, how many were real"
Recall    = TP / (TP + FN)   "of the real flaws, how many I caught"
F1        = harmonic mean of the two (one balanced score).

Matching is per flaw_type and greedy by best IoU, so one true event is credited to at
most one prediction and vice-versa (no double counting).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .intervals import interval_iou, start_end_error

Event = tuple[str, tuple[float, float]]  # (flaw_type, (start_s, end_s))


@dataclass
class MatchResult:
    """Outcome of matching one file's predictions against its ground truth."""
    tp: list[tuple[Event, Event, float]] = field(default_factory=list)  # (pred, gt, iou)
    fp: list[Event] = field(default_factory=list)                       # unmatched preds
    fn: list[Event] = field(default_factory=list)                       # unmatched gts


def match_events(preds: list[Event], gts: list[Event], iou_thr: float) -> MatchResult:
    """Greedily pair predictions to ground-truth events of the SAME flaw_type.

    Takes:
      preds   : list of (flaw_type, (start, end)) the detectors emitted (blind).
      gts     : list of (flaw_type, (start, end)) revealed from the manifest.
      iou_thr : minimum interval overlap (IoU) to count a pair as a match.
    Returns:
      MatchResult with true positives (pred, gt, iou), false positives, false negatives.
    Flow: run_eval calls this once per file, then sums TP/FP/FN across the TEST set.
    """
    # All candidate (pred_i, gt_j, iou) pairs that are same-class and clear the threshold.
    cand = []
    for i, (pf, pint) in enumerate(preds):
        for j, (gf, gint) in enumerate(gts):
            if pf != gf:
                continue
            iou = interval_iou(pint, gint)
            if iou >= iou_thr:
                cand.append((iou, i, j))
    cand.sort(reverse=True)  # best overlap first

    used_p: set[int] = set()
    used_g: set[int] = set()
    res = MatchResult()
    for iou, i, j in cand:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        res.tp.append((preds[i], gts[j], iou))
    res.fp = [p for i, p in enumerate(preds) if i not in used_p]
    res.fn = [g for j, g in enumerate(gts) if j not in used_g]
    return res


def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1 from raw counts. Zero-safe (0 when undefined)."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def localization_stats(tps: list[tuple[Event, Event, float]]) -> dict:
    """Mean IoU and mean absolute start/end error over matched (TP) events.

    Only matched events have a localization error (a miss has no interval to compare).
    Returns zeros/None-safe dict so an empty class does not crash the report.
    """
    if not tps:
        return {"n": 0, "mean_iou": 0.0, "mean_start_err_s": 0.0, "mean_end_err_s": 0.0}
    ious = [iou for _, _, iou in tps]
    se = [start_end_error(pred[1], gt[1]) for pred, gt, _ in tps]
    return {
        "n": len(tps),
        "mean_iou": sum(ious) / len(ious),
        "mean_start_err_s": sum(s for s, _ in se) / len(se),
        "mean_end_err_s": sum(e for _, e in se) / len(se),
    }


if __name__ == "__main__":
    # Self-check: a perfect hit, a false alarm, and a miss score as expected.
    preds = [("long_pause", (4.0, 6.0)),      # matches gt below (IoU 1.0) -> TP
             ("monotone", (1.0, 2.0))]         # no monotone gt -> FP
    gts = [("long_pause", (4.0, 6.0)),         # matched -> TP
           ("rate_deviation", (8.0, 9.0))]     # nothing predicts it -> FN
    r = match_events(preds, gts, iou_thr=0.3)
    assert len(r.tp) == 1 and len(r.fp) == 1 and len(r.fn) == 1, r
    p, rec, f = prf1(1, 1, 1)
    assert abs(p - 0.5) < 1e-9 and abs(rec - 0.5) < 1e-9 and abs(f - 0.5) < 1e-9, (p, rec, f)
    # Below-threshold overlap must NOT match (barely touching intervals).
    r2 = match_events([("long_pause", (4.0, 6.0))], [("long_pause", (5.9, 7.9))], 0.3)
    assert len(r2.tp) == 0 and len(r2.fp) == 1 and len(r2.fn) == 1, r2
    loc = localization_stats(r.tp)
    assert loc["n"] == 1 and abs(loc["mean_iou"] - 1.0) < 1e-9, loc
    print("metrics OK")
