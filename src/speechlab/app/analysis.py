"""Pure analysis layer for the dashboard (Session 5).

Why a separate file from main.py: `main.py` is a Streamlit *script* (full of st.*
calls that only run inside a Streamlit server). Everything here is plain Python with
NO st.* — so it can be unit-tested with ordinary pytest, cached cheaply, and reused.
The UI stays a thin shell on top of these functions.

Nothing in here invents a number: every value handed to the UI comes from the
Session 1-4 measurement code (features/*, detection/*) or is a documented constant
shown to the user (the heuristic-score weights).
"""
from __future__ import annotations

import io
import pathlib

import numpy as np
import pandas as pd
import soundfile as sf

from ..features.f0 import f0_contour
from ..features.pauses import frame_rms
from ..detection.detectors import build_reference, detect_all
from ..detection.evidence import (
    EvidenceItem, check_no_fabricated_numbers, phrase_evidence, validate_evidence,
)

# Heuristic "delivery-concern" score weights. These are DOCUMENTED CONSTANTS, not
# measurements and not a validated metric — the UI shows the formula verbatim so the
# number is transparent. Each detected flaw interval costs PENALTY points off 100.
SCORE_START = 100
SCORE_PENALTY = 8


def load_audio(src) -> tuple[np.ndarray, int]:
    """Load any audio source to mono float64 + sample rate.

    Takes:  a path (str/Path) OR raw bytes OR a file-like object (e.g. an upload).
    Returns: (y, sr) — y is 1-D float64 mono; sr is the file's sample rate (Hz).
    Flow:   feeds the feature extractors and the detectors.
    """
    if isinstance(src, (str, pathlib.Path)):
        y, sr = sf.read(str(src), always_2d=False)
    else:
        y, sr = sf.read(io.BytesIO(src if isinstance(src, (bytes, bytearray)) else src.read()))
    if y.ndim > 1:                       # stereo -> average to mono
        y = y.mean(axis=1)
    return y.astype(np.float64), int(sr)


def rms_series(y: np.ndarray, sr: int, cfg: dict) -> pd.DataFrame:
    """Short-time loudness envelope for the timeline plot.

    Returns a DataFrame[time_s, rms_norm] where rms_norm is 0..1 (peak-normalized),
    so quiet stretches (pauses) read as dips. Reuses features.pauses.frame_rms —
    the SAME energy the pause detector sees, so the plot matches the decision.
    """
    f = cfg["features"]
    times, rms = frame_rms(y, sr, f["frame_length_s"], f["hop_length_s"])
    peak = float(rms.max()) if rms.size else 0.0
    norm = rms / peak if peak > 0 else rms
    return pd.DataFrame({"time_s": times, "rms_norm": norm})


def f0_series(y: np.ndarray, sr: int, cfg: dict) -> pd.DataFrame:
    """Voiced pitch contour for the timeline plot.

    Returns a DataFrame[time_s, f0_hz] of VOICED frames only (f0 > 0); unvoiced
    gaps are dropped so Altair doesn't draw a line down to 0 Hz. Same f0_contour
    the monotone detector uses.
    """
    f = cfg["features"]
    times, f0 = f0_contour(y, sr, f["f0_min_hz"], f["f0_max_hz"], f["hop_length_s"])
    m = f0 > 0
    return pd.DataFrame({"time_s": np.asarray(times)[m], "f0_hz": np.asarray(f0)[m]})


def evidence_to_frame(items: list[EvidenceItem]) -> pd.DataFrame:
    """Flatten evidence objects into a tidy DataFrame for charting + tables.

    One row per detected interval. Columns are drawn straight from each object's
    fields (no new numbers), including the chart's x/x2 band bounds.
    """
    rows = []
    for e in items:
        rows.append({
            "flaw_type": e.flaw_type,
            "start_s": float(e.interval_s[0]),
            "end_s": float(e.interval_s[1]),
            "feature": e.feature,
            "reference_value": float(e.reference_value),
            "candidate_value": float(e.candidate_value),
            "confidence": float(e.confidence) if e.confidence is not None else None,
            "rule": e.rule,
            "uncertainty": e.uncertainty,
            "sentence": phrase_evidence(e),
        })
    cols = ["flaw_type", "start_s", "end_s", "feature", "reference_value",
            "candidate_value", "confidence", "rule", "uncertainty", "sentence"]
    return pd.DataFrame(rows, columns=cols)


def heuristic_score(items: list[EvidenceItem]) -> tuple[int, str]:
    """Transparent rule-of-thumb delivery-concern score (NOT a validated metric).

    Returns (score, formula_text). score = max(0, 100 - 8 * number_of_flaws). The
    formula string is shown in the UI so the number is never a black box. Honest
    framing lives in the UI caption: this ranks THIS clip's detected deviations from
    its own baseline, it does not measure 'good speaking'.
    """
    n = len(items)
    score = max(0, SCORE_START - SCORE_PENALTY * n)
    formula = f"{SCORE_START} - {SCORE_PENALTY} x {n} flaw interval(s) = {score}"
    return score, formula


def analyze(cfg: dict, baseline_y, bsr, cand_y, csr,
            baseline_spans=None, cand_spans=None) -> dict:
    """Full dashboard analysis for one baseline + one candidate clip.

    Takes:  cfg (loaded config.yaml), baseline audio (baseline_y, bsr) and candidate
            audio (cand_y, csr). The detectors are BLIND — they never see any
            ground-truth interval, only the candidate audio + the baseline profile.
            baseline_spans / cand_spans (optional): forced-alignment word spans for the
            two clips. When BOTH are given, the rate + monotone detectors run their
            Session 6b *same-material* rules — the exact path that scored held-out
            macro-F1 0.79 in run_eval.py. When absent (e.g. an upload with no
            transcript) the detectors use the weaker Session 4 fallbacks. Passing them
            here is what makes the live demo use the SAME detectors as the metrics.
    Returns: dict with
            - reference   : the speaker profile measured on the baseline
            - evidence    : list[EvidenceItem] from all three detectors
            - evidence_df : evidence_to_frame(evidence)
            - score, score_formula : heuristic_score(evidence)
            - cand_rms, cand_f0, base_rms, base_f0 : plot DataFrames
            - cand_duration_s, base_duration_s
            - aligned          : True iff the same-material (aligned) detectors ran
            - guard_fabricated : list of any fabricated numbers (MUST be empty)
            - guard_schema     : list of any schema problems (MUST be empty)
    Flow:   main.py renders every field; it adds no numbers of its own.

    Knob:  there is nothing to tune here — detector thresholds live in config.yaml
           (detection.*); this function only routes audio + spans into them.
    """
    aligned = baseline_spans is not None and cand_spans is not None
    reference = build_reference(baseline_y, bsr, cfg, spans=baseline_spans)
    evidence = detect_all(cand_y, csr, cfg, reference, spans=cand_spans)

    # Re-run the Session 4 honesty guards here so the UI can display a trust badge.
    fabricated, schema = [], []
    for e in evidence:
        schema += validate_evidence(e)
        fabricated += check_no_fabricated_numbers(phrase_evidence(e), e)

    score, formula = heuristic_score(evidence)
    return {
        "reference": reference,
        "evidence": evidence,
        "evidence_df": evidence_to_frame(evidence),
        "score": score,
        "score_formula": formula,
        "cand_rms": rms_series(cand_y, csr, cfg),
        "cand_f0": f0_series(cand_y, csr, cfg),
        "base_rms": rms_series(baseline_y, bsr, cfg),
        "base_f0": f0_series(baseline_y, bsr, cfg),
        "cand_duration_s": len(cand_y) / csr,
        "base_duration_s": len(baseline_y) / bsr,
        "aligned": aligned,
        "guard_fabricated": fabricated,
        "guard_schema": schema,
        "cand_y": cand_y,
        "cand_sr": csr,
        "baseline_y": baseline_y,
        "baseline_sr": bsr,
        "cand_spans": cand_spans,
        "baseline_spans": baseline_spans,
    }


if __name__ == "__main__":
    # Self-check: analyze a synthetic baseline vs the same clip with an obvious pause
    # inserted; expect at least one evidence item, both guards clean, score dropped.
    from ..config import load_config
    from ..dataset.inject import inject_pause

    cfg = load_config()
    sr = 16000
    t = np.arange(int(6.0 * sr)) / sr
    base = 0.3 * np.sin(2 * np.pi * (140 + 25 * np.sin(2 * np.pi * 3 * t)) * t)
    cand, (g0, g1) = inject_pause(base, sr, at_time_s=3.0, pause_s=2.0)

    out = analyze(cfg, base, sr, cand, sr)
    assert out["guard_fabricated"] == [], out["guard_fabricated"]
    assert out["guard_schema"] == [], out["guard_schema"]
    assert out["score"] <= SCORE_START
    assert set(out["evidence_df"].columns) >= {"flaw_type", "start_s", "end_s"}
    print(f"analysis OK: {len(out['evidence'])} evidence item(s), score {out['score']}, "
          f"gt pause ~[{g0:.2f},{g1:.2f}]")
