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
from ..features.pauses import detect_silences, frame_rms
from .evidence import EvidenceItem, compute_confidence


def build_reference(y: np.ndarray, sr: int, cfg: dict, spans=None) -> dict:
    """Measure the baseline once; these numbers are every detector's 'normal'.

    Returns a dict with: longest natural pause, fastest natural local rate, the typical
    (median) local pitch variability, the mean inter-word gap RMS (for filler detection),
    and which rate method was used. All speaker-specific.

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

    # Mean inter-word gap RMS: the natural energy level between words in this speaker's
    # baseline. Used by the filler_word detector — a filler has much MORE energy than
    # a natural breath or lip noise in the gap.
    mean_gap_rms = 0.0
    if spans is not None and len(spans) > 1:
        rms_times, rms_vals = frame_rms(y, sr, f["frame_length_s"], f["hop_length_s"])
        gap_energies = []
        for i in range(len(spans) - 1):
            g_start = spans[i].end_s
            g_end = spans[i + 1].start_s
            if g_end - g_start < 0.01:  # skip negligible gaps
                continue
            m = (rms_times >= g_start) & (rms_times < g_end)
            if m.sum() > 0:
                gap_energies.append(float(rms_vals[m].mean()))
        if gap_energies:
            mean_gap_rms = float(np.mean(gap_energies))

    return {"longest_pause_s": longest_pause, "max_local_rate_hz": max_rate,
            "baseline_spans": spans,
            "baseline_f0_times": times, "baseline_f0": f0,
            "typical_local_f0_std_hz": typical_f0_std,
            "mean_inter_word_rms": mean_gap_rms}


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
            scale = max(0.1, float(d.get("long_pause_excess_s", 0.3)))
            conf = compute_confidence(dur - thresh, scale)
            out.append(EvidenceItem(
                flaw_type="long_pause", interval_s=[round(s, 3), round(e, 3)],
                feature="pause_duration_s", reference_value=round(base, 3),
                candidate_value=round(dur, 3),
                rule=(f"pause {dur:.2f}s exceeds this speaker's longest natural pause "
                      f"{base:.2f}s by >= {d['long_pause_excess_s']:.2f}s"),
                confidence=conf,
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
            base_rate = r["base_rate_hz"]
            cand_rate = r["cand_rate_hz"]
            speedup = cand_rate / base_rate if base_rate > 0 else 1.0
            thresh_speedup = d["rate_speedup_ratio"]
            scale = max(0.1, thresh_speedup - 1.0)
            conf = compute_confidence(speedup - thresh_speedup, scale)
            out.append(EvidenceItem(
                flaw_type="rate_deviation",
                interval_s=[round(r["start_s"], 3), round(r["end_s"], 3)],
                feature="same_word_local_rate_hz",
                reference_value=round(base_rate, 2),
                candidate_value=round(cand_rate, 2),
                rule=(f"{r['n_words']} words spoken at {cand_rate:.2f}/s vs this "
                      f"speaker's {base_rate:.2f}/s for the SAME words "
                      f"(>= {pct:.0f}% faster)"),
                confidence=conf,
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
        if (e - s) < 0.5:
            continue  # ignore brief sub-0.5s transient spikes
        peak = max(r for a, b, r in hot if a >= s - 1e-6 and b <= e + 1e-6)
        scale = max(0.5, base * d["rate_deviation_ratio"])
        conf = compute_confidence(peak - thresh, scale)
        out.append(EvidenceItem(
            flaw_type="rate_deviation", interval_s=[round(s, 3), round(e, 3)],
            feature="local_syllable_rate_hz", reference_value=round(base, 2),
            candidate_value=round(peak, 2),
            rule=(f"local rate {peak:.2f}/s exceeds this speaker's fastest natural "
                  f"{base:.2f}/s by > {d['rate_deviation_ratio']*100:.0f}%"),
            confidence=conf,
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
            base_std = r["base_std_hz"]
            cand_std = r["cand_std_hz"]
            meas_ratio = cand_std / base_std if base_std > 0 else 0.0
            thresh_ratio = d["monotone_f0_std_ratio"]
            margin = max(0.0, thresh_ratio - meas_ratio)
            scale = max(0.05, thresh_ratio / 2.0)
            conf = compute_confidence(margin, scale)
            out.append(EvidenceItem(
                flaw_type="monotone",
                interval_s=[round(r["start_s"], 3), round(r["end_s"], 3)],
                feature="same_word_f0_std_hz",
                reference_value=round(base_std, 2),
                candidate_value=round(cand_std, 2),
                rule=(f"{r['n_words']} words with pitch std {cand_std:.2f}Hz vs this "
                      f"speaker's {base_std:.2f}Hz for the SAME words "
                      f"(<= {thresh_ratio:.2f}x)"),
                confidence=conf,
                uncertainty=("Pitch std from pooled voiced frames per word run; voiceless "
                             "or short words contribute little, and run edges are word-quantized.")))
        return out

    # Fallback: no transcript/alignment — compare each window to the speaker's typical std.
    base = ref["typical_local_f0_std_hz"]
    thresh = base * d["monotone_f0_std_ratio"]
    win = local_f0_std(times, f0, d["monotone_window_s"], d["monotone_hop_s"], min_voiced=15)
    cold = [(w["start_s"], w["end_s"], w["std_hz"]) for w in win if w["std_hz"] < thresh]
    out = []
    for s, e in _merge([(a, b) for a, b, _ in cold]):
        if (e - s) < 0.75:
            continue  # ignore isolated sub-0.75s unaccented dips
        low = min(v for a, b, v in cold if a >= s - 1e-6 and b <= e + 1e-6)
        margin = max(0.0, thresh - low)
        scale = max(1.0, thresh / 2.0)
        conf = compute_confidence(margin, scale)
        out.append(EvidenceItem(
            flaw_type="monotone", interval_s=[round(s, 3), round(e, 3)],
            feature="local_f0_std_hz", reference_value=round(base, 2),
            candidate_value=round(low, 2),
            rule=(f"local pitch std {low:.2f}Hz is below {d['monotone_f0_std_ratio']:.2f}x "
                  f"this speaker's typical {base:.2f}Hz"),
            confidence=conf,
            uncertainty=("Low pitch variation can be a deliberate style, not a flaw; "
                         "judged only against this speaker's baseline.")))
    return out


def detect_filler_word(y, sr, cfg, ref, spans=None) -> list[EvidenceItem]:
    """Flag inter-word gaps containing voiced energy — likely filler words ('um'/'uh').

    Filler words are short voiced sounds (nasal hums, hesitation markers) that a
    speaker inserts between words. They show as VOICED ENERGY in gaps that the
    transcript does not account for. The forced aligner maps each transcript word
    to a time range; the spaces between those ranges are inter-word gaps. A gap
    that is mostly silent is a natural pause (handled by long_pause). A gap that
    contains substantial voiced energy above the filler RMS threshold is flagged.

    When baseline spans (ref["baseline_spans"]) and candidate `spans` are both present,
    we use the aligned word gaps. The detector compares the candidate's inter-word gap
    energy to what the SAME gap in the baseline had — a filler raises it well above
    the baseline's natural inter-word energy.

    Without alignment this falls back to looking for voiced segments in detected
    silences (less reliable — we only have the energy proxy, no word boundaries).
    """
    f, d = cfg["features"], cfg["detection"]
    base_spans = ref.get("baseline_spans")
    filler_min_s = d.get("filler_min_duration_s", 0.15)
    filler_energy_ratio = d.get("filler_energy_ratio", 2.5)

    if spans is not None and base_spans is not None and len(spans) > 1:
        # Same-material path: compare inter-word gap energy between baseline and candidate.
        times, rms = frame_rms(y, sr, f["frame_length_s"], f["hop_length_s"])
        hop_s = f["hop_length_s"]
        out = []
        n = min(len(base_spans), len(spans))
        for i in range(n - 1):
            # Gap between word i's end and word i+1's start in the candidate.
            gap_start = spans[i].end_s
            gap_end = spans[i + 1].start_s
            gap_dur = gap_end - gap_start
            if gap_dur < filler_min_s:
                continue  # too short to be a filler

            # Measure RMS energy in the candidate's gap.
            m = (times >= gap_start) & (times < gap_end)
            if m.sum() < 2:
                continue
            gap_rms = float(rms[m].mean())

            # Measure RMS energy in the baseline's equivalent gap.
            base_gap_start = base_spans[i].end_s
            base_gap_end = base_spans[i + 1].start_s if i + 1 < len(base_spans) else base_spans[-1].end_s
            # Baseline gap energy: use the stored reference value from build_reference.
            base_gap_rms = ref.get("mean_inter_word_rms", 0.0)

            base_gap_dur = max(0.0, base_gap_end - base_gap_start)
            excess_dur = gap_dur - base_gap_dur

            # A filler has much MORE energy than a natural inter-word gap, OR
            # an expanded hesitation gap with substantial non-silent energy (> 0.03 RMS).
            fired = False
            if base_gap_rms > 0 and (gap_rms / base_gap_rms) >= filler_energy_ratio:
                fired = True
                energy_ratio = gap_rms / base_gap_rms
                margin = energy_ratio - filler_energy_ratio
                scale = max(1.0, filler_energy_ratio)
                conf = compute_confidence(margin, scale)
                rule_text = (f"gap between words {i}-{i+1} has RMS {gap_rms:.4f} vs "
                             f"baseline mean inter-word RMS {base_gap_rms:.4f} "
                             f"(>= {filler_energy_ratio:.1f}x)")
                feat = "inter_word_gap_rms"
                ref_val = round(base_gap_rms, 4)
                cand_val = round(gap_rms, 4)
            elif excess_dur >= filler_min_s and gap_rms >= 0.03:
                # Contrastive same-material gap extension with voiced hesitation energy
                fired = True
                margin = excess_dur - filler_min_s
                scale = max(0.2, filler_min_s)
                conf = compute_confidence(margin, scale)
                rule_text = (f"gap between words {i}-{i+1} is {gap_dur:.2f}s "
                             f"(+{excess_dur:.2f}s over baseline {base_gap_dur:.2f}s) "
                             f"with hesitation RMS {gap_rms:.4f}")
                feat = "gap_duration_s"
                ref_val = round(base_gap_dur, 3)
                cand_val = round(gap_dur, 3)

            if fired:
                out.append(EvidenceItem(
                    flaw_type="filler_word",
                    interval_s=[round(gap_start, 3), round(gap_end, 3)],
                    feature=feat,
                    reference_value=ref_val,
                    candidate_value=cand_val,
                    rule=rule_text,
                    confidence=conf,
                    uncertainty=("Voiced energy in an inter-word gap suggests a filler, "
                                 "but could also be a breath or lip noise."),
                    word_range=[i, i + 1]))
        return out

    # Fallback: no alignment — look for short voiced segments inside detected silences.
    # This is weak (same limitation as Session 4 rate/monotone fallbacks).
    return []  # No filler detection without alignment — honest fallback.


DETECTORS = {"long_pause": detect_long_pause,
             "rate_deviation": detect_rate_deviation,
             "monotone": detect_monotone,
             "filler_word": detect_filler_word}


def detect_all(y, sr, cfg, ref, spans=None) -> list[EvidenceItem]:
    """Run every detector blind on one candidate; return all evidence items.

    `spans` (optional forced-alignment word spans) is used by the rate, monotone,
    and filler_word detectors (Session 6b+ same-material rules). None keeps the
    Session 4 fallbacks (filler_word returns empty without alignment).
    """
    items = []
    items.extend(detect_long_pause(y, sr, cfg, ref))
    items.extend(detect_rate_deviation(y, sr, cfg, ref, spans=spans))
    items.extend(detect_monotone(y, sr, cfg, ref, spans=spans))
    items.extend(detect_filler_word(y, sr, cfg, ref, spans=spans))
    return items
