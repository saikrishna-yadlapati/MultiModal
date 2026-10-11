"""Transparent, speaker-normalized baseline detectors (Session 4).

Each detector is BLIND: it sees the candidate audio and a reference profile built
from THIS speaker's baseline — never the ground-truth interval. "Speaker-normalized"
means every judgement is relative to the same speaker's own baseline, so a naturally
deep voice or a naturally brisk talker is not mislabelled a flaw (context §4.2).

Three rules, one per MVP flaw class. They are deliberately simple threshold rules
(the honest rule-based floor, docs/EVALUATION.md) that emit a strict EvidenceItem
instead of a bare interval. Thresholds are config knobs (detection.*), tuned on the
train split and verified held-out in Session 6 — not tuned on test data.
"""
from __future__ import annotations

import numpy as np

from ..features.f0 import f0_contour
from ..features.local import local_f0_std, local_syllable_rate
from ..features.pauses import detect_silences
from .evidence import EvidenceItem


def build_reference(y: np.ndarray, sr: int, cfg: dict) -> dict:
    """Measure the baseline once; these numbers are every detector's 'normal'.

    Returns a dict with: longest natural pause, fastest natural local syllable rate,
    and the typical (median) local pitch variability. All speaker-specific.
    """
    f = cfg["features"]
    d = cfg["detection"]
    silences = detect_silences(y, sr, f["frame_length_s"], f["hop_length_s"],
                               f["silence_rel_db"], f["min_pause_s"])
    longest_pause = max((e - s for s, e in silences), default=0.0)

    rate_win = local_syllable_rate(y, sr, f["frame_length_s"], f["hop_length_s"],
                                   d["rate_window_s"], d["rate_hop_s"])
    max_rate = max((w["rate_hz"] for w in rate_win), default=0.0)

    times, f0 = f0_contour(y, sr, f["f0_min_hz"], f["f0_max_hz"], f["hop_length_s"])
    f0_win = local_f0_std(times, f0, d["monotone_window_s"], d["monotone_hop_s"])
    typical_f0_std = float(np.median([w["std_hz"] for w in f0_win])) if f0_win else 0.0

    return {"longest_pause_s": longest_pause, "max_local_rate_hz": max_rate,
            "typical_local_f0_std_hz": typical_f0_std}


def _merge(windows, gap_s: float = 0.01):
    """Merge touching/adjacent (start,end) windows into contiguous intervals."""
    if not windows:
        return []
    windows = sorted(windows)
    merged = [list(windows[0])]
    for s, e in windows[1:]:
        if s <= merged[-1][1] + gap_s:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(m) for m in merged]


def detect_long_pause(y, sr, cfg, ref) -> list[EvidenceItem]:
    """Flag candidate pauses longer than the speaker's longest natural pause + margin."""
    f, d = cfg["features"], cfg["detection"]
    silences = detect_silences(y, sr, f["frame_length_s"], f["hop_length_s"],
                               f["silence_rel_db"], f["min_pause_s"])
    base = ref["longest_pause_s"]
    thresh = base + d["long_pause_excess_s"]
    out = []
    for s, e in silences:
        dur = e - s
        if dur >= thresh:
            out.append(EvidenceItem(
                flaw_type="long_pause", interval_s=[round(s, 3), round(e, 3)],
                feature="pause_duration_s", reference_value=round(base, 3),
                candidate_value=round(dur, 3),
                rule=(f"pause {dur:.2f}s exceeds this speaker's longest natural pause "
                      f"{base:.2f}s by >= {d['long_pause_excess_s']:.2f}s"),
                uncertainty=("Pause length only; no forced alignment, so a naturally "
                             "shifted inter-sentence pause could land here.")))
    return out


def detect_rate_deviation(y, sr, cfg, ref) -> list[EvidenceItem]:
    """Flag windows whose local syllable rate is far above the speaker's fastest natural."""
    f, d = cfg["features"], cfg["detection"]
    base = ref["max_local_rate_hz"]
    thresh = base * (1.0 + d["rate_deviation_ratio"])
    win = local_syllable_rate(y, sr, f["frame_length_s"], f["hop_length_s"],
                              d["rate_window_s"], d["rate_hop_s"])
    hot = [(w["start_s"], w["end_s"], w["rate_hz"]) for w in win if w["rate_hz"] > thresh]
    out = []
    for s, e in _merge([(a, b) for a, b, _ in hot]):
        peak = max(r for a, b, r in hot if a >= s - 1e-6 and b <= e + 1e-6)
        out.append(EvidenceItem(
            flaw_type="rate_deviation", interval_s=[round(s, 3), round(e, 3)],
            feature="local_syllable_rate_hz", reference_value=round(base, 2),
            candidate_value=round(peak, 2),
            rule=(f"local rate {peak:.2f}/s exceeds this speaker's fastest natural "
                  f"{base:.2f}/s by > {d['rate_deviation_ratio']*100:.0f}%"),
            uncertainty=("Rate is an energy-peak syllable proxy, not phonetic; the "
                         "window edges are coarse and no alignment is used.")))
    return out


def detect_monotone(y, sr, cfg, ref) -> list[EvidenceItem]:
    """Flag windows whose local pitch variability is far below the speaker's typical."""
    f, d = cfg["features"], cfg["detection"]
    base = ref["typical_local_f0_std_hz"]
    thresh = base * d["monotone_f0_std_ratio"]
    times, f0 = f0_contour(y, sr, f["f0_min_hz"], f["f0_max_hz"], f["hop_length_s"])
    win = local_f0_std(times, f0, d["monotone_window_s"], d["monotone_hop_s"])
    cold = [(w["start_s"], w["end_s"], w["std_hz"]) for w in win if w["std_hz"] < thresh]
    out = []
    for s, e in _merge([(a, b) for a, b, _ in cold]):
        low = min(v for a, b, v in cold if a >= s - 1e-6 and b <= e + 1e-6)
        out.append(EvidenceItem(
            flaw_type="monotone", interval_s=[round(s, 3), round(e, 3)],
            feature="local_f0_std_hz", reference_value=round(base, 2),
            candidate_value=round(low, 2),
            rule=(f"local pitch std {low:.2f}Hz is below {d['monotone_f0_std_ratio']:.2f}x "
                  f"this speaker's typical {base:.2f}Hz"),
            uncertainty=("Low pitch variation can be a deliberate style, not a flaw; "
                         "judged only against this speaker's baseline.")))
    return out


DETECTORS = {"long_pause": detect_long_pause,
             "rate_deviation": detect_rate_deviation,
             "monotone": detect_monotone}


def detect_all(y, sr, cfg, ref) -> list[EvidenceItem]:
    """Run every detector blind on one candidate; return all evidence items."""
    items = []
    for fn in DETECTORS.values():
        items.extend(fn(y, sr, cfg, ref))
    return items
