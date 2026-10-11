"""On-disk cache for forced-alignment word spans (Session 6b).

Alignment runs a neural model (~10 s/clip on CPU). The held-out eval, and especially
threshold tuning, re-run detection on the same files many times, so we align each file
ONCE and reuse the saved word spans. This also makes metrics reproducible: identical
cached spans -> identical numbers across runs.

The cache key is the manifest sample_id (one WAV per id, deterministic). Delete a JSON
file (or the whole data/align_cache/ dir) to force re-alignment.
"""
from __future__ import annotations

import json
import pathlib

from .aligner import WordSpan, align_words


def cached_align(sample_id: str, y, sr: int, transcript: str,
                 cache_dir: pathlib.Path) -> list[WordSpan]:
    """Return word spans for one clip, aligning + caching on first call.

    Takes:
      sample_id : manifest id, used as the cache filename.
      y, sr     : audio samples + rate (passed straight to align_words on a miss).
      transcript: the known text to force-align.
      cache_dir : directory holding <sample_id>.json files (created if missing).
    Returns:
      list[WordSpan] (same type align_words returns), either loaded or freshly computed.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{sample_id}.json"
    if path.exists():
        data = json.loads(path.read_text())
        return [WordSpan(**d) for d in data]
    spans = align_words(y, sr, transcript)
    path.write_text(json.dumps([vars(s) for s in spans]))
    return spans
