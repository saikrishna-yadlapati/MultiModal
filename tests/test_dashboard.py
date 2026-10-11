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
from speechlab.dataset.manifest import read_manifest  # noqa: E402
from speechlab.alignment.subprocess_align import spans_for  # noqa: E402
from speechlab.app.analysis import (  # noqa: E402
    SCORE_PENALTY, SCORE_START, analyze, evidence_to_frame, heuristic_score,
)

MAIN = str(ROOT / "src" / "speechlab" / "app" / "main.py")
ALIGN_CACHE = ROOT / "data" / "align_cache"

# Features emitted ONLY by the aligned same-material rules vs the Session 4 fallbacks.
SAME_MATERIAL = {"same_word_local_rate_hz", "same_word_f0_std_hz"}
FALLBACK_ONLY = {"local_syllable_rate_hz", "local_f0_std_hz"}


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


# ---------------------------------------------------------------- aligned path (Session 7)
# Regression guard for the bug where the dashboard ran the WEAK detectors (no spans) while
# run_eval.py ran the aligned same-material ones. These assert the dashboard now feeds
# forced-alignment spans, so demo == measured truth.
import soundfile as sf  # noqa: E402


def _load_wav(path):
    y, sr = sf.read(str(path), always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return np.asarray(y, dtype=np.float64), int(sr)


def _manifest_rows():
    cfg = load_config()
    return read_manifest(str(ROOT / cfg["paths"]["manifest"]))


def test_spans_for_reads_cache_without_subprocess():
    # Every dataset clip is pre-cached, so spans_for must return spans from disk (no align).
    rows = _manifest_rows()
    r = next(x for x in rows if x.baseline_or_variant == "baseline")
    spans = spans_for(r.sample_id, str(ROOT / r.audio_path), r.transcript_text, ALIGN_CACHE)
    assert spans and all(hasattr(s, "start_s") and hasattr(s, "end_s") for s in spans)


def test_dashboard_runs_the_SAME_MATERIAL_detectors_when_spans_passed():
    # With baseline+candidate spans, analyze() must take the aligned path and the
    # rate/monotone evidence must carry the same-material feature, never the fallback.
    cfg = load_config()
    rows = _manifest_rows()
    baselines = {r.pair_id: r for r in rows if r.baseline_or_variant == "baseline"}
    variants = [r for r in rows
                if r.baseline_or_variant == "variant"
                and r.flaw_type in ("rate_deviation", "monotone")]

    saw_same_material = False
    for v in variants:
        b = baselines[v.pair_id]
        by, bsr = _load_wav(ROOT / b.audio_path)
        cy, csr = _load_wav(ROOT / v.audio_path)
        bspans = spans_for(b.sample_id, str(ROOT / b.audio_path), b.transcript_text, ALIGN_CACHE)
        cspans = spans_for(v.sample_id, str(ROOT / v.audio_path), v.transcript_text, ALIGN_CACHE)
        out = analyze(cfg, by, bsr, cy, csr, bspans, cspans)
        assert out["aligned"] is True
        for e in out["evidence"]:
            if e.flaw_type in ("rate_deviation", "monotone"):
                assert e.feature in SAME_MATERIAL, (v.sample_id, e.feature)
                assert e.feature not in FALLBACK_ONLY
                saw_same_material = True
    # Not vacuous: at least one clip actually fired a same-material rate/monotone item.
    assert saw_same_material, "no same-material rate/monotone evidence fired on any clip"


def test_without_spans_falls_back_to_weak_detectors():
    # Mirror image: no spans -> aligned False -> any rate/monotone item is the fallback rule.
    cfg = load_config()
    rows = _manifest_rows()
    baselines = {r.pair_id: r for r in rows if r.baseline_or_variant == "baseline"}
    v = next(r for r in rows if r.flaw_type == "monotone")
    b = baselines[v.pair_id]
    by, bsr = _load_wav(ROOT / b.audio_path)
    cy, csr = _load_wav(ROOT / v.audio_path)
    out = analyze(cfg, by, bsr, cy, csr)  # no spans
    assert out["aligned"] is False
    for e in out["evidence"]:
        if e.flaw_type in ("rate_deviation", "monotone"):
            assert e.feature in FALLBACK_ONLY


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
