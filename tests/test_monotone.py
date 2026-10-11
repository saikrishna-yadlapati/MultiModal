"""Pure self-check for same_material_monotone_regions (Session 6b) — synthetic, no model."""
import pathlib
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from speechlab.features.local import same_material_monotone_regions  # noqa: E402


@dataclass
class _Span:
    start_s: float
    end_s: float


def _spans(n_words=6, wdur=0.4):
    return [_Span(i * wdur, (i + 1) * wdur) for i in range(n_words)]


def test_flattened_run_flagged():
    # 6 words over 2.4 s. Baseline pitch oscillates everywhere (high std). Candidate is
    # flattened to a constant over words 2-4 (0.8-2.0 s), lively elsewhere.
    hop = 0.01
    times = np.arange(0, 2.4, hop)
    base_f0 = 150 + 30 * np.sin(2 * np.pi * 2 * times)        # always wobbling
    cand_f0 = base_f0.copy()
    flat = (times >= 0.8) & (times < 2.0)
    cand_f0[flat] = 150.0                                     # pitch collapsed to a line
    spans = _spans()
    regions = same_material_monotone_regions(times, base_f0, times, cand_f0,
                                             spans, spans, flat_ratio=0.5, min_words=3)
    assert len(regions) == 1, regions
    r = regions[0]
    assert r["cand_std_hz"] < r["base_std_hz"]               # flatter locally
    assert r["start_s"] >= 0.8 and r["end_s"] <= 2.0


def test_natural_pitch_no_region():
    hop = 0.01
    times = np.arange(0, 2.4, hop)
    f0 = 150 + 30 * np.sin(2 * np.pi * 2 * times)
    spans = _spans()
    # Same lively pitch in both -> ratio ~1 everywhere -> nothing flagged.
    assert same_material_monotone_regions(times, f0, times, f0, spans, spans, 0.5, 3) == []
