"""Pure self-check for aligned_local_rate (Session 6b) — synthetic spans, no model."""
import pathlib
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from speechlab.features.local import (  # noqa: E402
    aligned_local_rate, same_material_rate_regions,
)


@dataclass
class _Span:
    start_s: float
    end_s: float


def _chain(durs):
    """Lay word durations end-to-end into contiguous spans."""
    spans, t = [], 0.0
    for d in durs:
        spans.append(_Span(t, t + d))
        t += d
    return spans


def test_rushed_run_flagged_same_material():
    # 10 words at 0.4 s each; candidate compresses words 3-6 to 0.15 s (>2x faster).
    base = _chain([0.4] * 10)
    cand = _chain([0.4, 0.4, 0.4, 0.15, 0.15, 0.15, 0.15, 0.4, 0.4, 0.4])
    regions = same_material_rate_regions(base, cand, speedup_ratio=0.5, min_words=3)
    assert len(regions) == 1, regions
    r = regions[0]
    assert r["cand_rate_hz"] > r["base_rate_hz"]      # faster locally
    # The run localizes to the compressed stretch; the summed-window widens it by up to
    # (min_words-1) words at each edge, so allow that spillover.
    assert r["start_s"] >= 0.8 and r["end_s"] <= 2.3


def test_floor_word_does_not_break_run():
    # A tiny floor-pinned word (ratio ~1) sits inside a rushed stretch; summed-window
    # logic must still flag one run across it, not split into sub-min_words fragments.
    base = _chain([0.4, 0.4, 0.4, 0.02, 0.4, 0.4, 0.4])
    cand = _chain([0.4, 0.15, 0.15, 0.02, 0.15, 0.15, 0.4])
    regions = same_material_rate_regions(base, cand, speedup_ratio=0.5, min_words=3)
    assert len(regions) == 1, regions


def test_no_speedup_no_region():
    base = _chain([0.4] * 6)
    cand = _chain([0.4] * 6)
    assert same_material_rate_regions(base, cand, 0.5, 3) == []


def test_rushed_cluster_has_higher_rate():
    # 10 words evenly over 10 s (1 word/s), then 10 words packed into 1 s (10 word/s).
    spread = [_Span(i + 0.0, i + 0.3) for i in range(10)]           # centers ~0.15..9.15
    packed = [_Span(10.0 + i * 0.1, 10.0 + i * 0.1 + 0.05) for i in range(10)]  # ~10.0..11.0
    spans = spread + packed
    win = aligned_local_rate(spans, total_s=12.0, win_s=1.0, hop_s=0.5)
    rates = [w["rate_hz"] for w in win]
    # The packed second must produce a far higher local rate than the spread region.
    assert max(rates) >= 8.0, max(rates)
    early = max(w["rate_hz"] for w in win if w["end_s"] <= 9.0)
    assert early <= 3.0, early


def test_empty_spans_zero_rate():
    win = aligned_local_rate([], total_s=5.0, win_s=1.0, hop_s=0.5)
    assert win and all(w["rate_hz"] == 0.0 for w in win)
