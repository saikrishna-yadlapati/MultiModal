"""Session 1 vertical slice: one sample, end to end.

1. Load a WAV + transcript.
2. Extract pause, speech-rate, and F0 features; print a report.
3. Inject ONE known pause into a copy (ground truth kept).
4. Detect long pauses BLIND on the copy (detector never sees the injection time).
5. Score the predicted interval against ground truth (IoU + start/end error).

Usage:
    python scripts/run_slice.py data/raw/baseline_001.wav data/raw/baseline_001.txt
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speechlab.config import load_config  # noqa: E402
from speechlab.dataset.inject import inject_pause  # noqa: E402
from speechlab.detection.pause_flaw import detect_long_pauses  # noqa: E402
from speechlab.evaluation.intervals import interval_iou, start_end_error  # noqa: E402
from speechlab.features.f0 import f0_contour, f0_stats  # noqa: E402
from speechlab.features.pauses import detect_silences  # noqa: E402
from speechlab.features.rate import speech_rate_wpm, voiced_duration_s  # noqa: E402

# Where to inject and how long — fixed so the slice is deterministic.
# 4.5s sits mid-sentence (voiced) for this baseline, so the ground truth is an
# isolated silence not merged with a natural inter-sentence pause.
INJECT_AT_S = 4.5
INJECT_PAUSE_S = 1.5


def load_wav_mono(path: pathlib.Path):
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float64), sr


def extract(y, sr, cfg, n_words):
    f = cfg["features"]
    silences = detect_silences(
        y, sr, f["frame_length_s"], f["hop_length_s"],
        f["silence_rel_db"], f["min_pause_s"],
    )
    total_s = len(y) / sr
    voiced_s = voiced_duration_s(silences, total_s)
    wpm = speech_rate_wpm(n_words, voiced_s)
    _, f0 = f0_contour(y, sr, f["f0_min_hz"], f["f0_max_hz"], f["hop_length_s"])
    stats = f0_stats(f0)
    return {
        "total_s": total_s,
        "silences": silences,
        "voiced_s": voiced_s,
        "wpm": wpm,
        "f0": stats,
    }


def print_report(title, feats):
    print(f"\n=== {title} ===")
    print(f"  duration        : {feats['total_s']:.2f} s")
    print(f"  voiced duration : {feats['voiced_s']:.2f} s")
    print(f"  speech rate     : {feats['wpm']:.1f} words/min (global)")
    print(f"  F0 mean / std   : {feats['f0']['mean_hz']:.1f} / "
          f"{feats['f0']['std_hz']:.1f} Hz  "
          f"(voiced {feats['f0']['voiced_fraction']*100:.0f}%)")
    print(f"  pauses (>{feats.get('min_pause_s', 0.15):.2f}s): "
          f"{len(feats['silences'])}")
    for s, e in feats["silences"]:
        print(f"      [{s:6.2f} -> {e:6.2f}]  {e - s:.2f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("transcript")
    args = ap.parse_args()

    cfg = load_config()
    y, sr = load_wav_mono(pathlib.Path(args.wav))
    transcript = pathlib.Path(args.transcript).read_text(encoding="utf-8")
    n_words = len(transcript.split())
    print(f"loaded {args.wav}  sr={sr}  samples={len(y)}  words={n_words}")

    # --- baseline features ---
    base = extract(y, sr, cfg, n_words)
    base["min_pause_s"] = cfg["features"]["min_pause_s"]
    print_report("BASELINE features", base)

    long_base = detect_long_pauses(base["silences"], cfg["detection"]["long_pause_s"])
    print(f"\n  baseline long pauses (>= {cfg['detection']['long_pause_s']}s): "
          f"{len(long_base)}")

    # --- inject a known flaw ---
    cand_y, gt = inject_pause(y, sr, INJECT_AT_S, INJECT_PAUSE_S)
    var_dir = ROOT / "data" / "variants"
    var_dir.mkdir(parents=True, exist_ok=True)
    cand_path = var_dir / "baseline_001__pause.wav"
    sf.write(cand_path, cand_y, sr)
    print(f"\ninjected {INJECT_PAUSE_S}s pause at {INJECT_AT_S}s -> {cand_path}")
    print(f"GROUND TRUTH interval (hidden from detector): "
          f"[{gt[0]:.3f} -> {gt[1]:.3f}]")

    cand = extract(cand_y, sr, cfg, n_words)
    cand["min_pause_s"] = cfg["features"]["min_pause_s"]
    print_report("CANDIDATE features", cand)

    # --- BLIND detection ---
    long_cand = detect_long_pauses(cand["silences"], cfg["detection"]["long_pause_s"])
    print(f"\n=== BLIND DETECTION ===")
    print(f"  long pauses found (>= {cfg['detection']['long_pause_s']}s): "
          f"{len(long_cand)}")
    for s, e, d in long_cand:
        print(f"      [{s:6.2f} -> {e:6.2f}]  {d:.2f}s")

    if not long_cand:
        print("\nRESULT: FAIL — detector found no long pause.")
        return 1

    # Match the predicted interval that best overlaps ground truth.
    best = max(long_cand, key=lambda p: interval_iou((p[0], p[1]), gt))
    pred = (best[0], best[1])
    iou = interval_iou(pred, gt)
    se, ee = start_end_error(pred, gt)
    thr = cfg["evaluation"]["iou_match_threshold"]
    extra = len(long_cand) - 1

    print(f"\n  predicted interval : [{pred[0]:.3f} -> {pred[1]:.3f}]")
    print(f"  ground truth       : [{gt[0]:.3f} -> {gt[1]:.3f}]")
    print(f"  IoU                : {iou:.3f}  (match threshold {thr})")
    print(f"  start / end error  : {se*1000:.0f} ms / {ee*1000:.0f} ms")
    if extra:
        print(f"  other long pauses  : {extra} (potential false positives — report honestly)")

    ok = iou >= thr
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} — "
          f"blind detection {'localized' if ok else 'missed'} the injected pause "
          f"(IoU {iou:.2f} {'>=' if ok else '<'} {thr}).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
