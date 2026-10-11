"""Generate the controlled contrastive variant set + severity sweep (Session 3).

For the one baseline we have, produce a gradient of flawed copies per flaw class:
  long_pause (insert silence), rate_deviation (rush a segment), monotone (flatten pitch).
Each variant is DETERMINISTIC: the same `seed` in config.yaml reproduces identical audio
and identical ground-truth intervals. Every variant's GT interval + transform params are
written to data/manifest/manifest.csv so later detection can be scored honestly.

This supersedes the 2-row seed written by scripts/build_manifest.py: it rewrites the
whole manifest = baseline + every sweep variant.

Usage: python scripts/make_variants.py
"""
from __future__ import annotations

import pathlib
import random
import sys

import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speechlab.config import load_config  # noqa: E402
from speechlab.dataset.inject import inject_pause  # noqa: E402
from speechlab.dataset.manifest import (  # noqa: E402
    ManifestRow, make_pair_id, make_sample_id, validate_manifest, write_manifest,
)
from speechlab.dataset.variants import change_rate, flatten_pitch  # noqa: E402

SPEAKER = "sapi_default"
TX_ID = "tx001"
LICENSE = "self-generated (Windows SAPI TTS); team-owned, redistributable"


def _pick(rng, lo, hi):
    """Seeded location in [lo, hi], rounded to ms so the GT is stable and readable."""
    return round(rng.uniform(lo, hi), 3)


def main() -> int:
    cfg = load_config()
    v = cfg["variants"]
    feat = cfg["features"]
    raw = ROOT / "data" / "raw" / "baseline_001.wav"
    txt = ROOT / "data" / "raw" / "baseline_001.txt"
    if not raw.exists() or not txt.exists():
        print(f"missing baseline — run scripts/make_sample.py first")
        return 1

    y, sr = sf.read(raw, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float64)
    dur = len(y) / sr
    transcript = txt.read_text(encoding="utf-8").strip()

    pid = make_pair_id(SPEAKER, TX_ID)
    var_dir = ROOT / "data" / "variants"
    var_dir.mkdir(parents=True, exist_ok=True)

    rows = [ManifestRow(
        sample_id=make_sample_id(pid, "baseline"), pair_id=pid, speaker_id=SPEAKER,
        transcript_id=TX_ID, transcript_text=transcript,
        audio_path="data/raw/baseline_001.wav", sample_rate=sr,
        baseline_or_variant="baseline", flaw_type="clean",
        alignment_method="none", alignment_status="none",
        source_url="synthesized by scripts/make_sample.py", license=LICENSE,
        split="train", notes="Session 1 synthetic baseline; single speaker.",
    )]

    # One seeded RNG for the whole run => reproducible locations.
    rng = random.Random(cfg["seed"])
    seg = v["seg_len_s"]
    lo = v["region_min_s"]
    hi_seg = dur - v["region_pad_end_s"] - seg   # latest start that fits a segment
    hi_pt = dur - v["region_pad_end_s"]          # latest point for a pause insert

    # Draw one location per flaw type (deterministic given seed).
    pause_at = _pick(rng, lo, hi_pt)
    rate_start = _pick(rng, lo, hi_seg)
    mono_start = _pick(rng, lo, hi_seg)

    specs = []  # (flaw_type, severity, new_y, gt, method, params)
    for sev in v["long_pause"]["severities"]:
        pause_s = round(sev * v["long_pause"]["pause_s_at_sev1"], 3)
        ny, gt = inject_pause(y, sr, pause_at, pause_s)
        specs.append(("long_pause", sev, ny, gt, "insert_silence",
                      {"at_time_s": pause_at, "pause_s": pause_s, "fill": "zeros"}))
    for sev in v["rate_deviation"]["severities"]:
        factor = round(1.0 - sev * (1.0 - v["rate_deviation"]["factor_at_sev1"]), 4)
        ny, gt, params = change_rate(y, sr, rate_start, rate_start + seg, factor)
        specs.append(("rate_deviation", sev, ny, gt, "wsola", params))
    for sev in v["monotone"]["severities"]:
        k = round(1.0 - sev, 4)
        ny, gt, params = flatten_pitch(y, sr, mono_start, mono_start + seg, k,
                                       feat["f0_min_hz"], feat["f0_max_hz"])
        specs.append(("monotone", sev, ny, gt, "psola_pitch_flatten", params))

    for flaw, sev, ny, gt, method, params in specs:
        sid = make_sample_id(pid, "variant", flaw, sev)
        rel = f"data/variants/{sid}.wav"
        sf.write(ROOT / rel, ny, sr)
        rows.append(ManifestRow(
            sample_id=sid, pair_id=pid, speaker_id=SPEAKER, transcript_id=TX_ID,
            transcript_text=transcript, audio_path=rel, sample_rate=sr,
            baseline_or_variant="variant", flaw_type=flaw, severity=sev,
            gt_start_s=round(gt[0], 3), gt_end_s=round(gt[1], 3),
            transform_method=method, transform_params=params,
            alignment_method="none", alignment_status="none",
            source_url="derived from baseline_001 by scripts/make_variants.py",
            license=LICENSE, split="train",
            notes="Computer-injected flaw = known ground truth for this manipulation "
                  "only, not proof of natural human delivery.",
        ))

    problems = validate_manifest(rows)
    if problems:
        print("MANIFEST INVALID:")
        for p in problems:
            print("  -", p)
        return 1

    out = ROOT / cfg["paths"]["manifest"]
    write_manifest(rows, out)
    print(f"wrote {len(rows)} rows ({len(specs)} variants) -> {out}")
    print(f"  locations (seed {cfg['seed']}): pause@{pause_at}s  "
          f"rate@{rate_start}s  monotone@{mono_start}s  (seg {seg}s)")
    for flaw, sev, _, gt, method, _ in specs:
        print(f"  {flaw:<15} sev {sev:<4} GT [{gt[0]:6.3f} -> {gt[1]:6.3f}]  {method}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
