"""Segfault-safe forced alignment for the long-lived dashboard (Session 7).

Why this file exists
--------------------
`run_eval.py` feeds forced-alignment word spans into the detectors, which turns on
the *same-material* rate/monotone rules (the ones that scored macro-F1 0.79). The
Streamlit dashboard was NOT doing this (`app/analysis.py` called the detectors with
no spans), so it silently ran the Session 4 FALLBACK rules — the energy-peak rate
proxy (0/4) and the typical-median monotone rule (123 false positives). Demo and
metrics were running different detectors.

The obvious fix — "just call `align_words` in the dashboard" — is unsafe: the CPU
torch aligner **segfaults on repeated in-process alignment** (Session 6b note), and
Streamlit is one long-lived process, so a few clicks would crash the whole app.

So alignment here is resolved two ways, cheapest first:
  1. **Cache hit** (the normal demo path): read the span JSON that
     `scripts/make_dataset.py` / `run_eval.py` already wrote to data/align_cache/.
     Pure file read, no torch, no segfault. All 78 dataset clips are pre-cached.
  2. **Cache miss**: align in a FRESH subprocess that does exactly one alignment and
     exits — the same "one-alignment-per-process" workaround Session 6b used, so the
     segfault never touches the Streamlit process. The child writes the cache; we read it.

If both fail (e.g. offline, model not downloaded) we return ``None`` and the detectors
fall back to their Session 4 no-transcript rules. We never fabricate spans.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

from .aligner import WordSpan


def _load_cache(path: pathlib.Path) -> list[WordSpan] | None:
    """Read a <sample_id>.json span file into WordSpan objects, or None if absent/corrupt."""
    if not path.exists():
        return None
    try:
        return [WordSpan(**d) for d in json.loads(path.read_text())]
    except (ValueError, TypeError):
        return None


def spans_for(sample_id: str, audio_path: str, transcript: str,
              cache_dir: pathlib.Path) -> list[WordSpan] | None:
    """Return forced-alignment word spans for one clip, cache-first then subprocess.

    Takes:
      sample_id : manifest id; also the cache filename (<sample_id>.json).
      audio_path: WAV path, only needed on a cache miss (passed to the child).
      transcript: known text to align, only needed on a cache miss.
      cache_dir : data/align_cache/ directory holding the span JSONs.
    Returns:
      list[WordSpan] on success (cached OR freshly aligned), or None if alignment is
      unavailable — the caller then uses the Session 4 fallback detectors.
    Flow:
      spans flow into build_reference(spans=...) and detect_all(spans=...), which
      switches on the same-material rate/monotone rules that match the held-out eval.
    """
    path = cache_dir / f"{sample_id}.json"
    cached = _load_cache(path)
    if cached is not None:
        return cached

    # Cache miss: align in a throwaway subprocess (one alignment, then it exits).
    root = pathlib.Path(__file__).resolve().parents[2]  # repo src/.. root
    env = {**_os_environ(), "PYTHONPATH": str(root / "src")}
    try:
        subprocess.run(
            [sys.executable, "-m", "speechlab.alignment.subprocess_align",
             sample_id, str(audio_path), transcript, str(cache_dir)],
            cwd=str(root), env=env, timeout=180, check=True,
            capture_output=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None  # offline / model missing / crash -> caller falls back, no fabrication
    return _load_cache(path)


def _os_environ() -> dict:
    import os
    return dict(os.environ)


def _worker(argv: list[str]) -> int:
    """Child entrypoint: align ONE clip, write its cache JSON, exit. Never reused."""
    sample_id, audio_path, transcript, cache_dir = argv
    import numpy as np
    import soundfile as sf

    from .cache import cached_align  # reuse the exact caching used by the eval

    y, sr = sf.read(audio_path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    cached_align(sample_id, np.asarray(y, dtype=np.float64), int(sr),
                 transcript, pathlib.Path(cache_dir))
    return 0


if __name__ == "__main__":
    sys.exit(_worker(sys.argv[1:]))
