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
from ..features.local import (
    local_f0_std, local_syllable_rate, same_material_monotone_regions,
    same_material_rate_regions,
)
from ..features.pauses import detect_silences
from .evidence import EvidenceItem


def build_reference(y: np.ndarray, sr: int, cfg: dict, spans=None) -> dict:
    """Measure the baseline once; these numbers are every detector's 'normal'.

    Returns a dict with: longest natural pause, fastest natural local rate, the typical
    (median) local pitch variability, and which rate method was used. All speaker-specific.

    `spans` (optional): the baseline's forced-alignment word spans. When given, the
    rate detector works same-material (candidate words vs the SAME baseline words), so the
    baseline spans are stored on the profile. The fastest natural energy-proxy rate is also
    kept as a fallback for uploads with no transcript. The candidate detector picks the
    matching method — see detect_rate_deviation.
    """
    f = cfg["features"]
    d = cfg["detection"]
    silences = detect_silences(y, sr, f["frame_length_s"], f["hop_length_s"],
                               f["silence_rel_db"], f["min_pause_s"])
    longest_pause = max((e - s for s, e in silences), default=0.0)

    # Fallback energy-proxy fastest natural rate (used only when no transcript is available).
    rate_win = local_syllable_rate(y, sr, f["frame_length_s"], f["hop_length_s"],
                                   d["rate_window_s"], d["rate_hop_s"])
    max_rate = max((w["rate_hz"] for w in rate_win), default=0.0)

    times, f0 = f0_contour(y, sr, f["f0_min_hz"], f["f0_max_hz"], f["hop_length_s"])
    f0_win = local_f0_std(times, f0, d["monotone_window_s"], d["monotone_hop_s"])
    typical_f0_std = float(np.median([w["std_hz"] for w in f0_win])) if f0_win else 0.0

    return {"longest_pause_s": longest_pause, "max_local_rate_hz": max_rate,
            "baseline_spans": spans,
            "baseline_f0_times": times, "baseline_f0": f0,
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


def detect_rate_deviation(y, sr, cfg, ref, spans=None) -> list[EvidenceItem]:
    """Flag runs of words the candidate rushes relative to its OWN baseline rendition.

    Session 6b honest rule: baseline and candidate speak the SAME transcript, so each word
    lines up. A rushed (time-compressed) segment makes those words shorter in the candidate
    than the baseline said them. Comparing the SAME words (same-material) cancels natural
    speaker speed and is far more sensitive than the Session 4 energy proxy that scored 0/4.

    When baseline spans (ref["baseline_spans"]) and candidate `spans` are both present it
    uses that same-material comparison; otherwise (e.g. an upload with no transcript) it
    falls back to the energy-peak syllable proxy vs the speaker's fastest natural window.
    """
    f, d = cfg["features"], cfg["detection"]
    base_spans = ref.get("baseline_spans")
    if spans is not None and base_spans is not None:
        regions = same_material_rate_regions(base_spans, spans,
                                              d["rate_speedup_ratio"], d["rate_min_words"])
        pct = d["rate_speedup_ratio"] * 100
        out = []
        for r in regions:
            out.append(EvidenceItem(
                flaw_type="rate_deviation",
                interval_s=[round(r["start_s"], 3), round(r["end_s"], 3)],
                feature="same_word_local_rate_hz",
                reference_value=round(r["base_rate_hz"], 2),
                candidate_value=round(r["cand_rate_hz"], 2),
                rule=(f"{r['n_words']} words spoken at {r['cand_rate_hz']:.2f}/s vs this "
                      f"speaker's {r['base_rate_hz']:.2f}/s for the SAME words "
                      f"(>= {pct:.0f}% faster)"),
                uncertainty=("Rate from forced-alignment word durations; alignment can "
                             "smear word boundaries, and the run edges are word-quantized.")))
        return out

    # Fallback: no transcript/alignment — energy-peak syllable proxy vs fastest natural.
    base = ref["max_local_rate_hz"]
    thresh = base * (1.0 + d["rate_deviation_ratio"])
    win = local_syllable_rate(y, sr, f["frame_length_s"], f["hop_length_s"],
                              d["rate_window_s"], d["rate_hop_s"])
    unc = ("Rate is an energy-peak syllable proxy, not phonetic; the window edges "
           "are coarse and no alignment is used.")
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
            uncertainty=unc))
    return out


def detect_monotone(y, sr, cfg, ref, spans=None) -> list[EvidenceItem]:
    """Flag runs of words the candidate speaks far flatter than its OWN baseline said them.

    Session 6b honest rule: same-material, like the rate detector. Baseline and candidate
    speak the same transcript; a monotone (pitch-flattened) stretch has much less pitch
    movement across the SAME words in the candidate than the baseline. Comparing a word to
    itself cancels natural prosody — the Session 4 "below the speaker's typical variability"
    rule produced 123 false positives because natural speech has many genuinely flat windows.

    When baseline F0+spans (on ref) and candidate `spans` are present it uses that
    comparison; otherwise it falls back to the Session 4 typical-median window rule.
    """
    f, d = cfg["features"], cfg["detection"]
    times, f0 = f0_contour(y, sr, f["f0_min_hz"], f["f0_max_hz"], f["hop_length_s"])
    base_spans = ref.get("baseline_spans")
    base_times = ref.get("baseline_f0_times")
    base_f0 = ref.get("baseline_f0")
    if spans is not None and base_spans is not None and base_f0 is not None:
        regions = same_material_monotone_regions(
            base_times, base_f0, times, f0, base_spans, spans,
            d["monotone_f0_std_ratio"], d["monotone_min_words"])
        out = []
        for r in regions:
            out.append(EvidenceItem(
                flaw_type="monotone",
                interval_s=[round(r["start_s"], 3), round(r["end_s"], 3)],
                feature="same_word_f0_std_hz",
                reference_value=round(r["base_std_hz"], 2),
                candidate_value=round(r["cand_std_hz"], 2),
                rule=(f"{r['n_words']} words with pitch std {r['cand_std_hz']:.2f}Hz vs this "
                      f"speaker's {r['base_std_hz']:.2f}Hz for the SAME words "
                      f"(<= {d['monotone_f0_std_ratio']:.2f}x)"),
                uncertainty=("Pitch std from pooled voiced frames per word run; voiceless "
                             "or short words contribute little, and run edges are word-quantized.")))
        return out

    # Fallback: no transcript/alignment — compare each window to the speaker's typical std.
    base = ref["typical_local_f0_std_hz"]
    thresh = base * d["monotone_f0_std_ratio"]
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


def detect_all(y, sr, cfg, ref, spans=None) -> list[EvidenceItem]:
    """Run every detector blind on one candidate; return all evidence items.

    `spans` (optional forced-alignment word spans) is used by the rate and monotone
    detectors (Session 6b same-material rules). None keeps the Session 4 fallbacks.
    """
    items = []
    items.extend(detect_long_pause(y, sr, cfg, ref))
    items.extend(detect_rate_deviation(y, sr, cfg, ref, spans=spans))
    items.extend(detect_monotone(y, sr, cfg, ref, spans=spans))
    return items
