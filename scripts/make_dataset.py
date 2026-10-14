"""Build the multi-speaker, multi-transcript contrastive dataset (Session 6a).

SUPERSEDES scripts/make_variants.py (single speaker / single transcript). It walks the
plan in src/speechlab/dataset/plan.py: for every (speaker, transcript) pair it
  1. synthesizes a clean baseline WAV with that explicit SAPI voice, then
  2. runs the Session 3 severity sweep (long_pause / rate_deviation / monotone),
and writes ONE manifest with a speaker- AND transcript-disjoint train/test split.

Reproducibility: each pair gets its own seeded RNG (seed:pair_id), so flaw LOCATIONS are
deterministic per pair; the transforms themselves are bit-exact (Session 3). SAPI synth
is deterministic for a given voice+text+rate on this machine, but is NOT guaranteed
byte-identical across OS/voice-engine versions, so the reproducibility claim is scoped to
"same machine, same voices" (printed below as sha256s to verify).

Usage: python scripts/make_dataset.py
Needs: Windows SAPI voices named in plan.VOICES (verify with GetInstalledVoices()).
"""
from __future__ import annotations

import hashlib
import pathlib
import random
import subprocess
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
from speechlab.dataset.plan import TRANSCRIPTS, VOICES, planned_pairs  # noqa: E402
from speechlab.dataset.variants import change_rate, flatten_pitch, inject_filler  # noqa: E402

LICENSE = "self-generated (Windows SAPI TTS); team-owned, redistributable"


def synth_sapi(text: str, voice: str, out_wav: pathlib.Path) -> None:
    """Speak `text` with the named SAPI `voice` to `out_wav` (Windows only).

    SelectVoice picks the explicit engine voice so each speaker_id is a KNOWN,
    reproducible voice rather than the ambiguous system default. Rate 0 = normal.
    """
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"$s.SelectVoice('{voice}');"
        "$s.Rate = 0;"
        f"$s.SetOutputToWaveFile('{out_wav}');"
        f"$s.Speak(@'\n{text}\n'@);"
        "$s.Dispose()"
    )
    subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], check=True)


def _pick(rng, lo, hi):
    """Seeded location in [lo, hi], rounded to ms so the GT is stable and readable."""
    return round(rng.uniform(lo, hi), 3)


def _sweep(y, sr, dur, cfg, rng):
    """Session 3 severity sweep for ONE baseline. Returns list of
    (flaw_type, severity, new_y, gt, method, params). Locations drawn from `rng`.
    """
    v, feat = cfg["variants"], cfg["features"]
    seg = v["seg_len_s"]
    lo = v["region_min_s"]
    hi_seg = dur - v["region_pad_end_s"] - seg   # latest start that still fits a segment
    hi_pt = dur - v["region_pad_end_s"]          # latest point for a pause insert
    pause_at = _pick(rng, lo, hi_pt)
    rate_start = _pick(rng, lo, hi_seg)
    mono_start = _pick(rng, lo, hi_seg)

    specs = []
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
    if "filler_word" in v:
        filler_at = _pick(rng, lo, hi_pt)
        for sev in v["filler_word"]["severities"]:
            filler_s = round(sev * v["filler_word"]["filler_s_at_sev1"], 3)
            ny, gt, params = inject_filler(y, sr, filler_at, filler_s, severity=sev)
            specs.append(("filler_word", sev, ny, gt, "synth_nasal_filler", params))
    return specs


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def main() -> int:
    cfg = load_config()
    raw_dir = ROOT / "data" / "raw"
    var_dir = ROOT / "data" / "variants"
    raw_dir.mkdir(parents=True, exist_ok=True)
    var_dir.mkdir(parents=True, exist_ok=True)

    rows: list[ManifestRow] = []
    hashes: list[str] = []
    for spk, tx, split in planned_pairs():
        text = TRANSCRIPTS[tx]
        pid = make_pair_id(spk, tx)
        base_wav = raw_dir / f"{pid}__baseline.wav"
        synth_sapi(text, VOICES[spk], base_wav)

        y, sr = sf.read(base_wav, always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float64)
        dur = len(y) / sr

        rows.append(ManifestRow(
            sample_id=make_sample_id(pid, "baseline"), pair_id=pid, speaker_id=spk,
            transcript_id=tx, transcript_text=text,
            audio_path=f"data/raw/{pid}__baseline.wav", sample_rate=sr,
            baseline_or_variant="baseline", flaw_type="clean",
            alignment_method="none", alignment_status="none",
            source_url="synthesized by scripts/make_dataset.py", license=LICENSE,
            split=split, notes=f"SAPI voice {VOICES[spk]!r}; clean baseline.",
        ))
        hashes.append(f"{pid}__baseline  {_sha256(base_wav)}")

        rng = random.Random(f"{cfg['seed']}:{pid}")
        specs = _sweep(y, sr, dur, cfg, rng)
        for flaw, sev, ny, gt, method, params in specs:
            sid = make_sample_id(pid, "variant", flaw, sev)
            rel = f"data/variants/{sid}.wav"
            sf.write(ROOT / rel, ny, sr)
            hashes.append(f"{sid}  {_sha256(ROOT / rel)}")
            rows.append(ManifestRow(
                sample_id=sid, pair_id=pid, speaker_id=spk, transcript_id=tx,
                transcript_text=text, audio_path=rel, sample_rate=sr,
                baseline_or_variant="variant", flaw_type=flaw, severity=sev,
                gt_start_s=round(gt[0], 3), gt_end_s=round(gt[1], 3),
                transform_method=method, transform_params=params,
                alignment_method="none", alignment_status="none",
                source_url=f"derived from {pid}__baseline by scripts/make_dataset.py",
                license=LICENSE, split=split,
                notes="Computer-injected flaw = known ground truth for this "
                      "manipulation only, not proof of natural human delivery.",
            ))

    problems = validate_manifest(rows)
    if problems:
        print("MANIFEST INVALID:")
        for p in problems:
            print("  -", p)
        return 1

    out = ROOT / cfg["paths"]["manifest"]
    write_manifest(rows, out)

    n_base = sum(1 for r in rows if r.baseline_or_variant == "baseline")
    n_train = sum(1 for r in rows if r.split == "train")
    n_test = sum(1 for r in rows if r.split == "test")
    print(f"wrote {len(rows)} rows ({n_base} baselines + {len(rows) - n_base} variants) "
          f"-> {out}")
    print(f"  split: {n_train} train / {n_test} test  (speaker- and transcript-disjoint)")
    print("  sha256 (first 12 hex, for reproducibility checks):")
    for h in hashes:
        print("   ", h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
