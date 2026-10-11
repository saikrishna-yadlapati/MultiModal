"""Seed the dataset manifest from the real Session 1 sample + its injected variant.

Writes one truthful row per existing audio file to data/manifest/manifest.csv.
Numbers (sample rate, ground-truth interval) are READ/derived from the actual files
and the recorded injection params — nothing is fabricated. Re-run after adding samples.

Usage: python scripts/build_manifest.py
"""
from __future__ import annotations

import pathlib
import sys

import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speechlab.config import load_config  # noqa: E402
from speechlab.dataset.manifest import (  # noqa: E402
    ManifestRow, make_pair_id, make_sample_id,
    validate_manifest, write_manifest,
)

# Injection params that produced data/variants/baseline_001__pause.wav (run_slice.py).
INJECT_AT_S = 4.5
INJECT_PAUSE_S = 1.5


def main() -> int:
    cfg = load_config()
    raw = ROOT / "data" / "raw" / "baseline_001.wav"
    txt = ROOT / "data" / "raw" / "baseline_001.txt"
    var = ROOT / "data" / "variants" / "baseline_001__pause.wav"
    for f in (raw, txt, var):
        if not f.exists():
            print(f"missing {f} — run scripts/make_sample.py and scripts/run_slice.py first")
            return 1

    sr = sf.info(raw).samplerate
    transcript = txt.read_text(encoding="utf-8").strip()

    # Ground truth derived exactly as inject_pause rounds it, so it matches the file.
    idx = round(INJECT_AT_S * sr)
    n_sil = round(INJECT_PAUSE_S * sr)
    gt_start = idx / sr
    gt_end = (idx + n_sil) / sr

    speaker = "sapi_default"          # Windows SAPI default voice (one speaker so far)
    tx_id = "tx001"
    pid = make_pair_id(speaker, tx_id)
    lic = "self-generated (Windows SAPI TTS); team-owned, redistributable"

    baseline = ManifestRow(
        sample_id=make_sample_id(pid, "baseline"), pair_id=pid, speaker_id=speaker,
        transcript_id=tx_id, transcript_text=transcript,
        audio_path="data/raw/baseline_001.wav", sample_rate=sr,
        baseline_or_variant="baseline", flaw_type="clean",
        alignment_method="none", alignment_status="none",
        source_url="synthesized by scripts/make_sample.py", license=lic,
        split="train", notes="Session 1 synthetic baseline; single speaker.",
    )
    variant = ManifestRow(
        sample_id=make_sample_id(pid, "variant", "long_pause", 1.0), pair_id=pid,
        speaker_id=speaker, transcript_id=tx_id, transcript_text=transcript,
        audio_path="data/variants/baseline_001__pause.wav", sample_rate=sr,
        baseline_or_variant="variant", flaw_type="long_pause", severity=1.0,
        gt_start_s=gt_start, gt_end_s=gt_end, transform_method="insert_silence",
        transform_params={"at_time_s": INJECT_AT_S, "pause_s": INJECT_PAUSE_S,
                          "fill": "zeros"},
        alignment_method="none", alignment_status="none",
        source_url="derived from baseline_001 by scripts/run_slice.py", license=lic,
        split="train",
        notes="Computer-injected pause = known ground truth for this manipulation only, "
              "not proof of natural human pause behavior.",
    )

    rows = [baseline, variant]
    problems = validate_manifest(rows)
    if problems:
        print("MANIFEST INVALID:")
        for p in problems:
            print("  -", p)
        return 1

    out = ROOT / cfg["paths"]["manifest"]
    write_manifest(rows, out)
    print(f"wrote {len(rows)} rows -> {out}")
    print(f"  sr={sr}  gt_pause=[{gt_start:.3f} -> {gt_end:.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
