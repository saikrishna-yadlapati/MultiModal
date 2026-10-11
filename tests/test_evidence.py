"""Session 4 self-checks: evidence contract + blind detectors.

Pure/local only — parselmouth PSOLA runs offline, no model download, no SAPI.
Run: python -m pytest tests/test_evidence.py -q
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speechlab.config import load_config  # noqa: E402
from speechlab.dataset.inject import inject_pause  # noqa: E402
from speechlab.dataset.variants import change_rate, flatten_pitch  # noqa: E402
from speechlab.detection.detectors import (  # noqa: E402
    build_reference, detect_all, detect_long_pause, detect_monotone,
    detect_rate_deviation,
)
from speechlab.detection.evidence import (  # noqa: E402
    EvidenceItem, check_no_fabricated_numbers, phrase_evidence, validate_evidence,
)

SR = 16000
CFG = load_config()


# ---- evidence contract --------------------------------------------------

def test_valid_evidence_phrases_without_fabrication():
    ev = EvidenceItem("long_pause", [4.50, 6.00], "pause_duration_s", 0.90, 1.50,
                      "rule text", severity=0.75, uncertainty="limited for reasons")
    assert validate_evidence(ev) == []
    assert check_no_fabricated_numbers(phrase_evidence(ev), ev) == []


def test_guard_catches_invented_number():
    ev = EvidenceItem("monotone", [2.0, 3.0], "local_f0_std_hz", 20.0, 2.0, "r",
                      uncertainty="u")
    text = phrase_evidence(ev) + " Speaker was 95 percent nervous."
    assert check_no_fabricated_numbers(text, ev) == ["95"]


def test_schema_rejects_bad_interval_and_empty_uncertainty():
    assert validate_evidence(
        EvidenceItem("monotone", [3.0, 1.0], "f", 1.0, 0.0, "r", uncertainty="u")) != []
    assert validate_evidence(
        EvidenceItem("monotone", [1.0, 2.0], "f", 1.0, 0.0, "r", uncertainty="")) != []


# ---- detectors fire on the right synthetic flaw -------------------------

def _bursts(period_s, dur_s=6.0, f_hz=150.0):
    """Tone gated on/off every period_s -> an energy-peak train (syllable proxy)."""
    t = np.arange(int(dur_s * SR)) / SR
    gate = (np.sin(2 * np.pi * (1.0 / period_s) * t) > 0).astype(float)
    return 0.3 * np.sin(2 * np.pi * f_hz * t) * gate


def _wobble(dur_s=6.0):
    t = np.arange(int(dur_s * SR)) / SR
    inst = 150 + 12 * np.sin(2 * np.pi * 2 * t)
    return 0.3 * np.sin(2 * np.pi * np.cumsum(inst) / SR)


def test_long_pause_fires_on_injection_not_on_baseline():
    base = _bursts(0.4)
    ref = build_reference(base, SR, CFG)
    assert detect_long_pause(base, SR, CFG, ref) == []        # no false alarm on baseline
    cand, gt = inject_pause(base, SR, 3.0, 1.5)               # 1.5s pause mid-clip
    items = detect_long_pause(cand, SR, CFG, ref)
    assert items, "detector missed an injected 1.5s pause"
    best = max(items, key=lambda e: min(e.interval_s[1], gt[1]) - max(e.interval_s[0], gt[0]))
    assert best.interval_s[0] <= gt[0] + 0.3 and best.interval_s[1] >= gt[1] - 0.3


def test_rate_detector_fires_on_rushed_segment():
    base = _bursts(0.5, dur_s=8.0)                           # ~2 bumps/s, steady
    ref = build_reference(base, SR, CFG)
    # Compress a long (5s) stretch hard so a full 1s window lands inside it.
    cand, gt, _ = change_rate(base, SR, 1.0, 6.0, 0.3)
    items = detect_rate_deviation(cand, SR, CFG, ref)
    assert items, "detector missed a rushed segment"


def test_monotone_detector_fires_on_flattened_segment():
    base = _wobble()
    ref = build_reference(base, SR, CFG)
    assert ref["typical_local_f0_std_hz"] > 0
    cand, gt, _ = flatten_pitch(base, SR, 2.0, 4.0, 0.0,
                                CFG["features"]["f0_min_hz"], CFG["features"]["f0_max_hz"])
    items = detect_monotone(cand, SR, CFG, ref)
    assert items, "detector missed a flattened (monotone) segment"
    best = min(items, key=lambda e: e.candidate_value)
    assert best.interval_s[0] < 4.0 and best.interval_s[1] > 2.0  # overlaps the flat region


def test_detect_all_emits_valid_number_safe_evidence():
    base = _wobble()
    ref = build_reference(base, SR, CFG)
    cand, _ = inject_pause(base, SR, 3.0, 1.5)
    for e in detect_all(cand, SR, CFG, ref):
        assert validate_evidence(e) == [], e
        assert check_no_fabricated_numbers(phrase_evidence(e), e) == [], e
