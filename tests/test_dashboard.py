"""Session 5 self-checks: dashboard analysis layer + headless app boot.

Two kinds of test, deliberately separated (see references/testing.md):
  * Pure logic (analyze/heuristic_score/evidence_to_frame) -> plain pytest.
  * UI behavior (does the Streamlit script run, show metrics + evidence) ->
    st.testing.v1.AppTest, in-process and headless (no browser, no server).

Run: python -m pytest tests/test_dashboard.py -q
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speechlab.config import load_config  # noqa: E402
from speechlab.dataset.inject import inject_pause  # noqa: E402
from speechlab.app.analysis import (  # noqa: E402
    SCORE_PENALTY, SCORE_START, analyze, evidence_to_frame, heuristic_score,
)

MAIN = str(ROOT / "src" / "speechlab" / "app" / "main.py")


def _baseline_and_paused():
    """Synthetic 6 s tone + a 2 s silence injected at 3 s. Returns (base, cand, sr)."""
    sr = 16000
    t = np.arange(int(6.0 * sr)) / sr
    base = 0.3 * np.sin(2 * np.pi * (140 + 25 * np.sin(2 * np.pi * 3 * t)) * t)
    cand, _gt = inject_pause(base, sr, at_time_s=3.0, pause_s=2.0)
    return base, cand, sr


# ---------------------------------------------------------------- pure logic
def test_heuristic_score_is_the_documented_formula():
    # Score must be exactly max(0, 100 - 8 * n); nothing invented.
    assert heuristic_score([]) == (SCORE_START, f"{SCORE_START} - {SCORE_PENALTY} x 0 flaw interval(s) = {SCORE_START}")
    s, _ = heuristic_score([object()] * 3)
    assert s == SCORE_START - SCORE_PENALTY * 3
    # Floors at 0, never negative, even with absurd flaw counts.
    assert heuristic_score([object()] * 999)[0] == 0


def test_analyze_detects_injected_pause_and_passes_both_guards():
    cfg = load_config()
    base, cand, sr = _baseline_and_paused()
    out = analyze(cfg, base, sr, cand, sr)

    assert out["guard_fabricated"] == []          # no number in prose absent from evidence
    assert out["guard_schema"] == []              # every evidence object valid
    assert len(out["evidence"]) >= 1              # the obvious pause is caught
    assert any(e.flaw_type == "long_pause" for e in out["evidence"])
    assert out["score"] <= SCORE_START            # at least one flaw -> score dropped


def test_evidence_to_frame_only_copies_object_fields():
    # Honesty contract: the frame introduces no numbers beyond the evidence objects.
    cfg = load_config()
    base, cand, sr = _baseline_and_paused()
    out = analyze(cfg, base, sr, cand, sr)
    df = evidence_to_frame(out["evidence"])
    assert set(df.columns) >= {"flaw_type", "start_s", "end_s", "feature",
                               "reference_value", "candidate_value", "rule"}
    assert len(df) == len(out["evidence"])        # one row per interval, no extras


# ---------------------------------------------------------------- headless UI
def test_app_boots_and_shows_evidence():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(MAIN, default_timeout=30).run()
    assert not at.exception                       # script ran clean end-to-end

    # Default candidate is the long_pause clip -> 4 KPI metrics render.
    assert len(at.metric) == 4
    labels = [m.label for m in at.metric]
    assert "Delivery-concern score" in labels

    # Trust badge: guards clean -> success, not error.
    assert at.success
    assert not at.error

    # "Evidence" subheader is always rendered.
    assert any("Evidence" in h.value for h in at.subheader)
