"""Session 4: run the blind, speaker-normalized detectors over the variant set.

Flow:
  1. Read the manifest; load the baseline and build ONE reference profile from it.
  2. For every variant, run all three detectors BLIND (they never see gt_start/gt_end).
  3. Print each EvidenceItem, its number-safe phrasing, and the fabrication-guard result.
  4. Only AFTER detecting, reveal the manifest ground truth and report, per flaw class
     and severity, whether the right detector fired and how well it localized.

No fabricated numbers: every value printed comes from measurement code or the manifest.

Usage: python scripts/run_detect.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speechlab.config import load_config  # noqa: E402
from speechlab.dataset.manifest import read_manifest  # noqa: E402
from speechlab.detection.detectors import DETECTORS, build_reference  # noqa: E402
from speechlab.detection.evidence import (  # noqa: E402
    check_no_fabricated_numbers, phrase_evidence, validate_evidence,
)
from speechlab.evaluation.intervals import interval_iou, start_end_error  # noqa: E402


def load_wav_mono(path: pathlib.Path):
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float64), sr


def main() -> int:
    cfg = load_config()
    manifest = read_manifest(ROOT / cfg["paths"]["manifest"])
    baseline = next((r for r in manifest if r.baseline_or_variant == "baseline"), None)
    variants = [r for r in manifest if r.baseline_or_variant == "variant"]
    if baseline is None or not variants:
        print("manifest needs a baseline + variants — run scripts/make_variants.py first")
        return 1

    by, bsr = load_wav_mono(ROOT / baseline.audio_path)
    ref = build_reference(by, bsr, cfg)
    print("REFERENCE profile (measured on baseline, this speaker):")
    print(f"  longest natural pause   : {ref['longest_pause_s']:.2f} s")
    print(f"  fastest local rate      : {ref['max_local_rate_hz']:.2f} syll/s")
    print(f"  typical local F0 std    : {ref['typical_local_f0_std_hz']:.2f} Hz")

    thr = cfg["evaluation"]["iou_match_threshold"]
    rows = []  # (flaw, severity, fired_right_class, overlaps_gt, best_iou)
    guard_fail = 0

    for v in sorted(variants, key=lambda r: (r.flaw_type, r.severity or 0)):
        y, sr = load_wav_mono(ROOT / v.audio_path)
        print(f"\n=== {v.sample_id}  (flaw={v.flaw_type} sev={v.severity}) ===")

        fired_right, overlaps, best_iou = False, False, 0.0
        for name, fn in DETECTORS.items():
            items = fn(y, sr, cfg, ref)  # BLIND: no gt passed in
            for e in items:
                problems = validate_evidence(e)
                prose = phrase_evidence(e)
                bad = check_no_fabricated_numbers(prose, e)
                tag = "OK" if not problems and not bad else "INVALID"
                if problems or bad:
                    guard_fail += 1
                print(f"  [{name}] {tag}  interval [{e.interval_s[0]:.2f}->{e.interval_s[1]:.2f}]")
                print(f"      {prose}")
                if problems:
                    print(f"      !! schema problems: {problems}")
                if bad:
                    print(f"      !! fabricated numbers: {bad}")
                # scoring (reveal GT only here, for reporting — not given to detector)
                if name == v.flaw_type:
                    fired_right = True
                    gt = (v.gt_start_s, v.gt_end_s)
                    iou = interval_iou((e.interval_s[0], e.interval_s[1]), gt)
                    best_iou = max(best_iou, iou)
                    if iou > 0:
                        overlaps = True
        rows.append((v.flaw_type, v.severity, fired_right, overlaps, best_iou))

    # --- reveal + honest detection-vs-severity table ---
    print("\n\n=== DETECTION vs GROUND TRUTH (revealed after blind run) ===")
    print(f"{'flaw':<16}{'sev':>5}{'right class?':>14}{'overlaps GT?':>14}{'best IoU':>10}")
    for flaw, sev, fired, ov, iou in rows:
        print(f"{flaw:<16}{sev:>5}{('yes' if fired else 'NO'):>14}"
              f"{('yes' if ov else 'no'):>14}{iou:>10.3f}")

    n = len(rows)
    hit = sum(1 for *_, fired, ov, _ in [(*r,) for r in rows] if fired and ov)
    print(f"\nright-class AND overlapping GT: {hit}/{n} variants  (IoU match thr {thr})")
    print(f"evidence objects failing schema/number-guard: {guard_fail} (must be 0)")
    print("\nNote: pause/rate transforms shift later timestamps, so absolute-time IoU")
    print("understates localization; rigorous metrics (with alignment) come in Session 6.")
    return 0 if guard_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
