"""Session 2 alignment test: force-align the real baseline sample, print word times.

Verifies the torchaudio MMS_FA aligner on ONE sample and sanity-checks the output
(monotonic, within audio duration). First run downloads the model (needs internet once).

Usage: python scripts/run_align.py data/raw/baseline_001.wav data/raw/baseline_001.txt
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speechlab.alignment.aligner import align_words  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("transcript")
    args = ap.parse_args()

    y, sr = sf.read(args.wav, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float64)
    dur = len(y) / sr
    transcript = pathlib.Path(args.transcript).read_text(encoding="utf-8")

    print(f"loaded {args.wav}  sr={sr}  dur={dur:.2f}s")
    try:
        spans = align_words(y, sr, transcript)
    except RuntimeError as e:
        print(f"ALIGNMENT FAILED: {e}")
        print("-> fallback (energy-based word spread) would be used; see DECISIONS.md")
        return 1

    print(f"\naligned {len(spans)} words:")
    for s in spans:
        print(f"  [{s.start_s:6.2f} -> {s.end_s:6.2f}]  {s.score:.2f}  {s.word}")

    # Sanity checks (not a quality metric, just "did it produce sane timestamps").
    ok = True
    prev = -1.0
    for s in spans:
        if not (0 <= s.start_s <= s.end_s <= dur + 0.05):
            print(f"  BAD span out of [0,{dur:.2f}]: {s.word} [{s.start_s}->{s.end_s}]")
            ok = False
        if s.start_s < prev - 0.05:
            print(f"  NON-MONOTONIC at {s.word}: starts {s.start_s} < prev {prev}")
            ok = False
        prev = s.start_s

    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} — "
          f"{len(spans)} word timestamps, monotonic & within audio duration."
          if ok else f"\nRESULT: FAIL — see bad spans above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
