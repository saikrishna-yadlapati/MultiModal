"""Session 6b: held-out evaluation harness — real metrics on the TEST split only.

Flow:
  1. Read the manifest, keep ONLY split == "test" (an unseen speaker on unseen words).
  2. For each pair, build the speaker reference from THAT pair's own baseline.
  3. Run all three detectors BLIND on every test file (variants + clean baselines).
  4. Reveal ground truth and compute, per flaw class: precision / recall / F1,
     localization error (IoU, start/end), the false-positive rate on clean speech,
     and the detection-vs-severity sweep.

Honesty: thresholds come from config.yaml (NOT tuned on this test set); every number
printed comes from measurement code or the manifest; the no-fabrication guard is run
over every evidence object and the failure count is reported (must be 0).

Usage: python scripts/run_eval.py
"""
from __future__ import annotations

import pathlib
import sys
from collections import defaultdict

import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speechlab.config import load_config  # noqa: E402
from speechlab.alignment.cache import cached_align  # noqa: E402
from speechlab.dataset.manifest import read_manifest  # noqa: E402
from speechlab.detection.detectors import build_reference, detect_all  # noqa: E402
from speechlab.detection.evidence import (  # noqa: E402
    check_no_fabricated_numbers, phrase_evidence, validate_evidence,
)
from speechlab.evaluation.metrics import (  # noqa: E402
    localization_stats, match_events, prf1,
)

FLAWS = ("long_pause", "rate_deviation", "monotone")


def load_wav_mono(path: pathlib.Path):
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float64), sr


def main() -> int:
    cfg = load_config()
    iou_thr = cfg["evaluation"]["iou_match_threshold"]
    manifest = read_manifest(ROOT / cfg["paths"]["manifest"])
    test = [r for r in manifest if r.split == "test"]
    if not test:
        print("no test rows — run scripts/make_dataset.py first")
        return 1

    # One reference per pair, measured on that pair's own baseline (speaker-matched).
    # Forced-align every clip once (cached) so local rate is real words-per-second.
    cache_dir = ROOT / "data" / "align_cache"
    baselines = {r.pair_id: r for r in test if r.baseline_or_variant == "baseline"}
    refs: dict[str, dict] = {}
    for pid, b in baselines.items():
        by, bsr = load_wav_mono(ROOT / b.audio_path)
        bspans = cached_align(b.sample_id, by, bsr, b.transcript_text, cache_dir)
        refs[pid] = build_reference(by, bsr, cfg, spans=bspans)

    # Per-class tallies, matched events (for localization), severity sweep, guard count.
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    matched = defaultdict(list)
    sweep = defaultdict(lambda: [0, 0])  # (flaw, sev) -> [detected, total]
    clean_events = 0; clean_seconds = 0.0
    guard_fail = 0

    for v in sorted(test, key=lambda r: (r.pair_id, r.flaw_type, r.severity or 0)):
        y, sr = load_wav_mono(ROOT / v.audio_path)
        ref = refs.get(v.pair_id)
        if ref is None:
            print(f"!! no baseline for pair {v.pair_id}; skipping {v.sample_id}")
            continue

        spans = cached_align(v.sample_id, y, sr, v.transcript_text, cache_dir)
        items = detect_all(y, sr, cfg, ref, spans=spans)  # BLIND: ground truth never passed in
        for e in items:
            if validate_evidence(e) or check_no_fabricated_numbers(phrase_evidence(e), e):
                guard_fail += 1
        preds = [(e.flaw_type, (e.interval_s[0], e.interval_s[1])) for e in items]

        if v.baseline_or_variant == "baseline":  # clean speech -> any event is a false alarm
            clean_events += len(preds)
            clean_seconds += len(y) / sr
            res = match_events(preds, [], iou_thr)
        else:
            gt = [(v.flaw_type, (v.gt_start_s, v.gt_end_s))]
            res = match_events(preds, gt, iou_thr)
            det = 1 if res.tp else 0
            s = sweep[(v.flaw_type, v.severity)]
            s[0] += det; s[1] += 1

        for cls in FLAWS:
            tp[cls] += sum(1 for p, _, _ in res.tp if p[0] == cls)
            fp[cls] += sum(1 for p in res.fp if p[0] == cls)
            fn[cls] += sum(1 for g in res.fn if g[0] == cls)
            matched[cls].extend(t for t in res.tp if t[0][0] == cls)

    # ---------- report ----------
    print("=== HELD-OUT EVALUATION (test split: unseen speaker + unseen transcripts) ===")
    print(f"thresholds from config.yaml (NOT tuned on this set); IoU match >= {iou_thr}\n")

    print(f"{'flaw':<16}{'TP':>4}{'FP':>4}{'FN':>4}{'prec':>8}{'rec':>8}{'F1':>8}")
    macro = []
    for cls in FLAWS:
        p, r, f = prf1(tp[cls], fp[cls], fn[cls])
        macro.append(f)
        print(f"{cls:<16}{tp[cls]:>4}{fp[cls]:>4}{fn[cls]:>4}{p:>8.2f}{r:>8.2f}{f:>8.2f}")
    print(f"{'macro-F1':<16}{'':>12}{'':>8}{'':>8}{sum(macro)/len(macro):>8.2f}")

    print("\n--- localization error on matched events (true positives) ---")
    print(f"{'flaw':<16}{'n':>4}{'mean IoU':>10}{'start err s':>13}{'end err s':>12}")
    for cls in FLAWS:
        L = localization_stats(matched[cls])
        print(f"{cls:<16}{L['n']:>4}{L['mean_iou']:>10.3f}"
              f"{L['mean_start_err_s']:>13.3f}{L['mean_end_err_s']:>12.3f}")

    fp_per_min = clean_events / (clean_seconds / 60.0) if clean_seconds else 0.0
    print(f"\n--- false alarms on CLEAN speech ---")
    print(f"{clean_events} events over {clean_seconds:.1f}s clean "
          f"= {fp_per_min:.2f} false events / minute")

    print("\n--- detection rate vs severity (right class AND overlaps GT) ---")
    print(f"{'flaw':<16}" + "".join(f"{s:>8}" for s in (0.25, 0.5, 0.75, 1.0)))
    for cls in FLAWS:
        cells = []
        for sev in (0.25, 0.5, 0.75, 1.0):
            det, tot = sweep.get((cls, sev), [0, 0])
            cells.append(f"{det}/{tot}" if tot else "-")
        print(f"{cls:<16}" + "".join(f"{c:>8}" for c in cells))

    print(f"\nevidence objects failing schema/number-guard: {guard_fail} (must be 0)")
    return 0 if guard_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
