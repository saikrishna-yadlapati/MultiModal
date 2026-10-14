"""Ingest a real human speaker baseline and generate contrastive variants (Session 8c).

Dataset provenance:
  - Source: Lab41-SRI VOiCES Corpus (Speaker sp0307, Utterance ch127535-sg0042)
  - License: Creative Commons Attribution 4.0 International (CC-BY 4.0)
  - Audio: 16 kHz studio-quality clean human speech, male speaker.
  - Transcript (tx005): "I had that curiosity beside me at this moment."
  - Split: "val" (preserves disjoint train/test split and held-out test benchmarks).

Generates:
  1. data/raw/human_sp0307__tx005__baseline.wav (clean human baseline)
  2. data/variants/human_sp0307__tx005__long_pause_s1.wav
  3. data/variants/human_sp0307__tx005__rate_deviation_s1.wav
  4. data/variants/human_sp0307__tx005__monotone_s1.wav
  5. data/variants/human_sp0307__tx005__filler_word_s1.wav
  6. data/align_cache/*.json forced-alignment span caches for each clip.
  7. Appends 5 rows to data/manifest/manifest.csv and validates schema + leakage.

Usage: python scripts/ingest_human_baseline.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import urllib.request
import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speechlab.alignment.aligner import align_words
from speechlab.dataset.inject import inject_pause
from speechlab.dataset.manifest import (
    ManifestRow, make_pair_id, make_sample_id, read_manifest,
    validate_manifest, write_manifest,
)
from speechlab.dataset.variants import change_rate, flatten_pitch, inject_filler

VOICES_URL = "https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
LICENSE = "Creative Commons Attribution 4.0 International (CC-BY 4.0); Lab41-SRI VOiCES"
TRANSCRIPT = "I had that curiosity beside me at this moment."
SPEAKER_ID = "human_sp0307"
TRANSCRIPT_ID = "tx005"
PAIR_ID = make_pair_id(SPEAKER_ID, TRANSCRIPT_ID)
SPLIT = "val"


def main():
    raw_dir = ROOT / "data" / "raw"
    var_dir = ROOT / "data" / "variants"
    cache_dir = ROOT / "data" / "align_cache"
    manifest_path = ROOT / "data" / "manifest" / "manifest.csv"

    raw_dir.mkdir(parents=True, exist_ok=True)
    var_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    base_wav_path = raw_dir / f"{PAIR_ID}__baseline.wav"
    temp_path = raw_dir / "temp_voices.wav"

    if temp_path.exists():
        y, sr = sf.read(temp_path)
    else:
        print(f"Downloading human speech sample from {VOICES_URL}...")
        urllib.request.urlretrieve(VOICES_URL, base_wav_path)
        y, sr = sf.read(base_wav_path)

    # Save canonical baseline WAV (16 kHz mono float)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float64)
    sf.write(base_wav_path, y, sr)

    print(f"1. Human baseline saved: {base_wav_path} ({len(y)/sr:.2f}s, {sr}Hz)")

    # Align baseline and cache spans
    base_sid = make_sample_id(PAIR_ID, "baseline")
    base_spans = align_words(y, sr, TRANSCRIPT)
    (cache_dir / f"{base_sid}.json").write_text(
        json.dumps([{"word": s.word, "start_s": s.start_s, "end_s": s.end_s,
                     "score": s.score, "orig_index": s.orig_index} for s in base_spans], indent=2)
    )
    print(f"   Aligned baseline: {len(base_spans)} words -> {cache_dir / f'{base_sid}.json'}")

    # Generate contrastive variants
    variants_to_build = []

    # 1. Long pause (inject 1.5s silence at 1.80s, right between 'curiosity' and 'beside')
    pause_y, pause_gt = inject_pause(y, sr, at_time_s=1.80, pause_s=1.50)
    variants_to_build.append((
        "long_pause", 1.0, pause_y, pause_gt, "insert_silence",
        {"at_time_s": 1.80, "pause_s": 1.50, "fill": "zeros"}
    ))

    # 2. Rate deviation (speed up 'curiosity beside' ~0.9s..2.3s by factor 0.55)
    rate_y, rate_gt, rate_params = change_rate(y, sr, start_s=0.90, end_s=2.30, factor=0.55)
    variants_to_build.append((
        "rate_deviation", 1.0, rate_y, rate_gt, "wsola", rate_params
    ))

    # 3. Monotone (flatten pitch of 'curiosity beside' ~0.85s..2.30s, k=0.0)
    mono_y, mono_gt, mono_params = flatten_pitch(y, sr, start_s=0.85, end_s=2.30, k=0.0,
                                                 fmin=65.0, fmax=400.0)
    variants_to_build.append((
        "monotone", 1.0, mono_y, mono_gt, "psola_pitch_flatten", mono_params
    ))

    # 4. Filler word (inject 0.45s nasal filler at gap ~1.82s)
    filler_y, filler_gt, filler_params = inject_filler(y, sr, at_time_s=1.82, filler_s=0.45, severity=1.0)
    variants_to_build.append((
        "filler_word", 1.0, filler_y, filler_gt, "synth_nasal_filler", filler_params
    ))

    new_rows: list[ManifestRow] = [
        ManifestRow(
            sample_id=base_sid, pair_id=PAIR_ID, speaker_id=SPEAKER_ID,
            transcript_id=TRANSCRIPT_ID, transcript_text=TRANSCRIPT,
            audio_path=f"data/raw/{base_wav_path.name}", sample_rate=sr,
            baseline_or_variant="baseline", flaw_type="clean",
            alignment_method="mms_fa", alignment_status="ok",
            source_url=VOICES_URL, license=LICENSE, split=SPLIT,
            notes="Natural human speech baseline from Lab41-SRI VOiCES corpus (CC-BY 4.0). Studio recording.",
        )
    ]

    for flaw, sev, vy, gt, method, params in variants_to_build:
        vid = make_sample_id(PAIR_ID, "variant", flaw, sev)
        v_path = var_dir / f"{vid}.wav"
        sf.write(v_path, vy, sr)

        # Align variant and cache
        v_spans = align_words(vy, sr, TRANSCRIPT)
        (cache_dir / f"{vid}.json").write_text(
            json.dumps([{"word": s.word, "start_s": s.start_s, "end_s": s.end_s,
                         "score": s.score, "orig_index": s.orig_index} for s in v_spans], indent=2)
        )
        print(f"   Generated {flaw} variant -> {v_path.name} (GT: {gt[0]:.2f}s-{gt[1]:.2f}s)")

        new_rows.append(ManifestRow(
            sample_id=vid, pair_id=PAIR_ID, speaker_id=SPEAKER_ID,
            transcript_id=TRANSCRIPT_ID, transcript_text=TRANSCRIPT,
            audio_path=f"data/variants/{v_path.name}", sample_rate=sr,
            baseline_or_variant="variant", flaw_type=flaw, severity=sev,
            gt_start_s=round(gt[0], 3), gt_end_s=round(gt[1], 3),
            transform_method=method, transform_params=params,
            alignment_method="mms_fa", alignment_status="ok",
            source_url=f"derived from {base_sid} by scripts/ingest_human_baseline.py",
            license=LICENSE, split=SPLIT,
            notes="Natural human voice with controlled acoustic flaw for external validity demonstration.",
        ))

    # Read existing manifest, deduplicate if already added, append and write
    existing_rows = read_manifest(manifest_path)
    filtered = [r for r in existing_rows if r.pair_id != PAIR_ID]
    all_rows = filtered + new_rows

    problems = validate_manifest(all_rows)
    if problems:
        print("Manifest validation errors:")
        for p in problems:
            print("  *", p)
        return 1

    write_manifest(all_rows, manifest_path)
    print(f"\nManifest successfully updated with {len(new_rows)} human rows (Total: {len(all_rows)} rows).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
