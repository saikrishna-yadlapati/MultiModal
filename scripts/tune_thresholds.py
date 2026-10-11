"""Session 6b-4: tune detection thresholds on the TRAIN split ONLY.

Honesty rule (docs/EVALUATION.md, CLAUDE.md): thresholds may be chosen on train, NEVER on
test. This script sweeps each flaw's knobs over a small grid, scores F1 on the train split,
and prints the best value. The owner copies the winners into config.yaml by hand; the held-
out numbers in scripts/run_eval.py are then produced with thresholds this script never saw
the test set choose.

Scoring per flaw F:
  positives = train variants with flaw_type == F (GT interval known)
  negatives = every other train row (clean baselines + other-flaw variants) -> any firing
              of detector F on them is a false positive.
Audio, alignment and per-pair references are built ONCE; the grid only re-runs the (cheap)
detector with a mutated config, so the sweep is fast.

Usage: python scripts/tune_thresholds.py
"""
from __future__ import annotations

import copy
import pathlib
import sys

import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speechlab.config import load_config  # noqa: E402
from speechlab.alignment.cache import cached_align  # noqa: E402
from speechlab.dataset.manifest import read_manifest  # noqa: E402
from speechlab.detection.detectors import (  # noqa: E402
    build_reference, detect_long_pause, detect_monotone, detect_rate_deviation,
)
from speechlab.evaluation.metrics import match_events, prf1  # noqa: E402

DETECT = {"long_pause": detect_long_pause,
          "rate_deviation": detect_rate_deviation,
          "monotone": detect_monotone}


def load_wav_mono(path: pathlib.Path):
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float64), sr


def score(cfg, flaw, rows, cache, refs, iou_thr):
    """TP/FP/FN then P/R/F1 for one flaw at the given config, over the train rows."""
    det = DETECT[flaw]
    tp = fp = fn = 0
    for r in rows:
        y, sr, spans = cache[r.sample_id]
        ref = refs[r.pair_id]
        if flaw == "long_pause":
            items = det(y, sr, cfg, ref)                 # long_pause takes no spans
        else:
            items = det(y, sr, cfg, ref, spans=spans)
        preds = [(e.flaw_type, (e.interval_s[0], e.interval_s[1])) for e in items]
        is_pos = (r.baseline_or_variant == "variant" and r.flaw_type == flaw)
        gt = [(flaw, (r.gt_start_s, r.gt_end_s))] if is_pos else []
        res = match_events(preds, gt, iou_thr)
        tp += len(res.tp); fp += len(res.fp); fn += len(res.fn)
    return (tp, fp, fn, *prf1(tp, fp, fn))


def sweep(cfg, flaw, grid, rows, cache, refs, iou_thr):
    """grid: list of (label, {knob: value}). Returns rows sorted best-F1-first."""
    results = []
    for label, overrides in grid:
        c = copy.deepcopy(cfg)
        c["detection"].update(overrides)
        tp, fp, fn, p, r, f = score(c, flaw, rows, cache, refs, iou_thr)
        results.append((f, p, tp, fp, fn, label, overrides))
    # best F1, tie-break on precision (fewer false alarms), then fewer FP.
    results.sort(key=lambda t: (t[0], t[1], -t[3]), reverse=True)
    return results


def main() -> int:
    cfg = load_config()
    iou_thr = cfg["evaluation"]["iou_match_threshold"]
    manifest = read_manifest(ROOT / cfg["paths"]["manifest"])
    train = [r for r in manifest if r.split == "train"]
    if not train:
        print("no train rows — run scripts/make_dataset.py first")
        return 1

    cache_dir = ROOT / "data" / "align_cache"
    cache = {}
    for r in train:
        y, sr = load_wav_mono(ROOT / r.audio_path)
        spans = cached_align(r.sample_id, y, sr, r.transcript_text, cache_dir)
        cache[r.sample_id] = (y, sr, spans)

    baselines = {r.pair_id: r for r in train if r.baseline_or_variant == "baseline"}
    refs = {pid: build_reference(*cache[b.sample_id][:2], cfg, spans=cache[b.sample_id][2])
            for pid, b in baselines.items()}

    grids = {
        "long_pause": [(f"excess={v}", {"long_pause_excess_s": v})
                       for v in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)],
        "rate_deviation": [(f"speedup={s},min_words={w}",
                            {"rate_speedup_ratio": s, "rate_min_words": w})
                           for s in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5) for w in (2, 3, 4)],
        "monotone": [(f"ratio={s},min_words={w}",
                      {"monotone_f0_std_ratio": s, "monotone_min_words": w})
                     for s in (0.4, 0.5, 0.6, 0.65, 0.7) for w in (2, 3)],
    }

    print("=== THRESHOLD TUNING on TRAIN split only (never test) ===")
    print(f"IoU match >= {iou_thr}; negatives = clean + other-flaw variants\n")
    for flaw, grid in grids.items():
        ranked = sweep(cfg, flaw, grid, train, cache, refs, iou_thr)
        print(f"--- {flaw} (best first) ---")
        print(f"{'setting':<28}{'F1':>7}{'prec':>7}{'TP':>5}{'FP':>5}{'FN':>5}")
        for f, p, tp, fp, fn, label, _ in ranked[:6]:
            print(f"{label:<28}{f:>7.2f}{p:>7.2f}{tp:>5}{fp:>5}{fn:>5}")
        best = ranked[0]
        print(f"  -> WINNER: {best[5]}  (F1 {best[0]:.2f}, prec {best[1]:.2f})\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
