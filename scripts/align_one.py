"""Align ONE clip by sample_id and cache it, then exit. Run one-per-process to dodge the
CPU torch forced-alignment segfault that appears when many alignments share a process.
Usage: python scripts/align_one.py <sample_id>
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speechlab.config import load_config  # noqa: E402
from speechlab.alignment.cache import cached_align  # noqa: E402
from speechlab.dataset.manifest import read_manifest  # noqa: E402


def main() -> int:
    sid = sys.argv[1]
    cfg = load_config()
    man = read_manifest(ROOT / cfg["paths"]["manifest"])
    r = next(x for x in man if x.sample_id == sid)
    y, sr = sf.read(ROOT / r.audio_path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    cached_align(sid, y.astype(np.float64), sr, r.transcript_text,
                 ROOT / "data" / "align_cache")
    print("OK", sid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
