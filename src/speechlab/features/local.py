"""Local (windowed) features — the per-moment versions of the global features.

Session 1 measured whole-clip F0 std and global words-per-minute. A flaw lives in
a short stretch, so to *localize* it we need the same measurements computed over a
sliding window: a small time box that slides along the clip, giving one number per
box. "Sliding window" = like dragging a 1-second ruler across the audio and writing
down a reading every quarter second.

Two helpers:
  - local_f0_std      : pitch variability inside each window (for the monotone detector)
  - local_syllable_rate: a syllable-per-second proxy from the energy envelope
                         (for the rate detector; no forced alignment needed)

All pure numpy + the existing f0 helper. No model download.
"""
from __future__ import annotations

import numpy as np

from .pauses import frame_rms


def _windows(total_s: float, win_s: float, hop_s: float):
    """Yield (start_s, end_s, center_s) boxes tiling [0, total_s]."""
    if total_s <= 0:
        return
    start = 0.0
    while start < total_s:
        end = min(start + win_s, total_s)
        yield start, end, (start + end) / 2.0
        if end >= total_s:
            break
        start += hop_s


def local_f0_std(times: np.ndarray, f0: np.ndarray, win_s: float, hop_s: float,
                 min_voiced: int = 5):
    """Std of voiced pitch inside each sliding window.

    Takes:
      times, f0 : outputs of features.f0.f0_contour (frame centers, Hz; 0 == unvoiced).
      win_s     : window length in seconds (config detection.monotone_window_s).
      hop_s     : how far the window slides each step (config detection.monotone_hop_s).
      min_voiced: skip windows with fewer than this many voiced frames (no pitch to judge).
    Returns:
      list of dicts {start_s, end_s, center_s, std_hz, n_voiced}, voiced windows only.
    Flow: the monotone detector scans these for the window with the LOWEST std_hz.
    """
    times = np.asarray(times)
    f0 = np.asarray(f0)
    total_s = float(times[-1]) if times.size else 0.0
    out = []
    for s, e, c in _windows(total_s, win_s, hop_s):
        m = (times >= s) & (times < e) & (f0 > 0)
        n = int(m.sum())
        if n < min_voiced:
            continue
        out.append({"start_s": s, "end_s": e, "center_s": c,
                    "std_hz": float(f0[m].std()), "n_voiced": n})
    return out


def same_material_rate_regions(base_spans, cand_spans, speedup_ratio: float,
                                min_words: int):
    """Find runs of words the candidate speaks much FASTER than its own baseline said them.

    This is the Session 6b honest rate detector. Baseline and candidate share the exact
    transcript, so word i is the SAME word in both; `orig_index` lines them up. A rushed
    (time-compressed) segment makes those words shorter in the candidate than in the
    baseline. Comparing the SAME words removes natural speaker speed differences — far
    more sensitive than comparing against the speaker's single fastest natural window.

    We judge a RUN, not single words: forced alignment pins short function words ("a",
    "for") to a floor duration, so their candidate/baseline ratio stays ~1.0 even inside
    a rushed stretch and would break a per-word run. Instead we slide a `min_words`-wide
    window and compare the SUMMED durations (total candidate time vs total baseline time
    for the same words); a floor-pinned word barely moves a multi-word sum.

    Takes:
      base_spans, cand_spans : WordSpan lists for the baseline and candidate (same text).
      speedup_ratio : flag a window whose summed candidate duration is this fraction
                      shorter, i.e. sum(cand)/sum(base) <= 1/(1+speedup_ratio).
                      0.5 => the same words spoken >=33% faster. (config detection.rate_speedup_ratio)
      min_words     : window width in words (config detection.rate_min_words); also the
                      smallest run that can be flagged, so one noisy word is ignored.
    Returns:
      list of dicts {start_s, end_s, n_words, cand_rate_hz, base_rate_hz} in CANDIDATE time.
    Flow: the rate detector turns each region into an EvidenceItem; the interval is the
    candidate-timeline span of the run, the two rates are its same-material evidence.
    """
    n = min(len(base_spans), len(cand_spans))
    w = max(1, min_words)
    dur_thresh = 1.0 / (1.0 + speedup_ratio)
    bd = [base_spans[i].end_s - base_spans[i].start_s for i in range(n)]
    cd = [cand_spans[i].end_s - cand_spans[i].start_s for i in range(n)]

    flagged = [False] * n
    for i in range(0, n - w + 1):
        bsum = sum(bd[i:i + w])
        csum = sum(cd[i:i + w])
        if bsum > 0 and (csum / bsum) <= dur_thresh:
            for k in range(i, i + w):
                flagged[k] = True

    out = []
    i = 0
    while i < n:
        if not flagged[i]:
            i += 1
            continue
        j = i
        while j < n and flagged[j]:
            j += 1
        run = list(range(i, j))
        cs, ce = cand_spans[run[0]].start_s, cand_spans[run[-1]].end_s
        bs, be = base_spans[run[0]].start_s, base_spans[run[-1]].end_s
        cdur, bdur = ce - cs, be - bs
        out.append({
            "start_s": cs, "end_s": ce, "n_words": len(run),
            "cand_rate_hz": len(run) / cdur if cdur > 0 else 0.0,
            "base_rate_hz": len(run) / bdur if bdur > 0 else 0.0,
        })
        i = j
    return out


def _pooled_f0_std(times: np.ndarray, f0: np.ndarray, t0: float, t1: float,
                   min_voiced: int):
    """Std of voiced pitch pooled over [t0, t1); None if too few voiced frames."""
    m = (times >= t0) & (times < t1) & (f0 > 0)
    vals = f0[m]
    return float(vals.std()) if vals.size >= min_voiced else None


def same_material_monotone_regions(base_times, base_f0, cand_times, cand_f0,
                                   base_spans, cand_spans, flat_ratio: float,
                                   min_words: int, min_voiced: int = 5):
    """Find runs of words whose pitch the candidate FLATTENS vs its own baseline.

    The pitch-variability twin of same_material_rate_regions. Baseline and candidate speak
    the same transcript, so word i is the same word in both. A monotone (pitch-flattened)
    segment has far LESS pitch movement across those words in the candidate than the
    baseline had. Comparing the SAME words to themselves cancels natural prosody — the
    Session 4 detector compared each window to the speaker's typical (median) variability
    and drowned in false positives (123 of them), because natural speech is full of
    genuinely low-variation windows (steady vowels, sentence-final falls).

    We pool voiced F0 over a sliding `min_words`-wide word window and compare the candidate's
    pitch std to the baseline's over the SAME words. Pooling across several words captures
    the intonation CONTOUR (the rise/fall of a phrase); flattening collapses it. A single
    naturally-flat word is flat in BOTH, so its ratio stays ~1 and it is not flagged.

    Takes:
      base_times, base_f0 : baseline f0_contour output (frame centers, Hz; 0 == unvoiced).
      cand_times, cand_f0 : candidate f0_contour output.
      base_spans, cand_spans : WordSpan lists (same transcript, aligned by index).
      flat_ratio : flag a window whose candidate pooled std <= this * baseline pooled std
                   for the same words. 0.5 => half the pitch variation. (monotone_f0_std_ratio)
      min_words  : window width in words; also the smallest run flagged (monotone_min_words).
      min_voiced : skip a window with fewer than this many voiced frames on either side.
    Returns:
      list of dicts {start_s, end_s, n_words, cand_std_hz, base_std_hz} in CANDIDATE time.
    """
    n = min(len(base_spans), len(cand_spans))
    w = max(1, min_words)
    flagged = [False] * n
    for i in range(0, n - w + 1):
        bstd = _pooled_f0_std(base_times, base_f0,
                              base_spans[i].start_s, base_spans[i + w - 1].end_s, min_voiced)
        cstd = _pooled_f0_std(cand_times, cand_f0,
                              cand_spans[i].start_s, cand_spans[i + w - 1].end_s, min_voiced)
        if bstd and bstd > 0 and cstd is not None and (cstd / bstd) <= flat_ratio:
            for k in range(i, i + w):
                flagged[k] = True

    out = []
    i = 0
    while i < n:
        if not flagged[i]:
            i += 1
            continue
        j = i
        while j < n and flagged[j]:
            j += 1
        run = list(range(i, j))
        cs, ce = cand_spans[run[0]].start_s, cand_spans[run[-1]].end_s
        bs, be = base_spans[run[0]].start_s, base_spans[run[-1]].end_s
        cstd = _pooled_f0_std(cand_times, cand_f0, cs, ce, min_voiced) or 0.0
        bstd = _pooled_f0_std(base_times, base_f0, bs, be, min_voiced) or 0.0
        out.append({"start_s": cs, "end_s": ce, "n_words": len(run),
                    "cand_std_hz": cstd, "base_std_hz": bstd})
        i = j
    return out


def aligned_local_rate(spans, total_s: float, win_s: float, hop_s: float):
    """Words-per-second inside each sliding window, from forced-alignment word spans.

    This is the Session 6b replacement for the energy-peak syllable proxy below. A
    "word span" is one word's [start,end] in seconds, produced by the aligner (karaoke
    timing). Counting word centers per window gives a real local speech rate: a rushed
    (time-compressed) segment packs the same words into less time, so its rate spikes.

    Takes:
      spans   : list of objects with .start_s / .end_s (speechlab.alignment WordSpan).
      total_s : clip duration in seconds (window grid extent).
      win_s   : window length (config detection.rate_window_s).
      hop_s   : window step (config detection.rate_hop_s).
    Returns:
      list of dicts {start_s, end_s, center_s, rate_hz, n_words}, one per window.
    Flow: build_reference takes the MAX of these on the baseline; the rate detector
    flags candidate windows far above that speaker-specific fastest natural rate.
    """
    centers = [(w.start_s + w.end_s) / 2.0 for w in spans]
    out = []
    for s, e, c in _windows(total_s, win_s, hop_s):
        n = sum(1 for cc in centers if s <= cc < e)
        dur = e - s
        out.append({"start_s": s, "end_s": e, "center_s": c,
                    "rate_hz": n / dur if dur > 0 else 0.0, "n_words": n})
    return out


def _envelope_peaks(times: np.ndarray, rms: np.ndarray,
                    rel_thresh: float = 0.15, min_sep_s: float = 0.12):
    """Peak times in the energy envelope ~ syllable nuclei (loud vowel centers).

    A syllable is roughly one energy bump (a vowel) between quieter consonants, so
    counting envelope peaks per second approximates syllables per second. This is a
    PROXY, not phonetics.
    ponytail: naive energy-peak syllable proxy; swap for a real syllable nuclei
    detector (e.g. Praat intensity + De Jong's method) if rate accuracy matters.
    """
    env = rms / (rms.max() + 1e-10)
    # 3-tap moving average to suppress tiny ripples before peak-picking.
    if env.size >= 3:
        env = np.convolve(env, np.ones(3) / 3.0, mode="same")
    hop_s = float(times[1] - times[0]) if times.size > 1 else min_sep_s
    min_sep = max(1, int(round(min_sep_s / hop_s)))
    peaks = []
    last = -min_sep
    for i in range(1, env.size - 1):
        if env[i] > rel_thresh and env[i] > env[i - 1] and env[i] >= env[i + 1]:
            if i - last >= min_sep:
                peaks.append(times[i])
                last = i
    return np.asarray(peaks)


def local_syllable_rate(y: np.ndarray, sr: int, frame_s: float, hop_s: float,
                        win_s: float, win_hop_s: float):
    """Syllables-per-second proxy inside each sliding window.

    Takes:
      y, sr            : audio and sample rate.
      frame_s, hop_s   : short-time RMS frame/hop (config features.*), for the envelope.
      win_s, win_hop_s : the sliding window length/step (config detection.rate_window_s/_hop_s).
    Returns:
      list of dicts {start_s, end_s, center_s, rate_hz, n_peaks}.
    Flow: the rate detector compares each candidate window's rate_hz to the baseline's
    fastest natural window; a window far above it is a 'rushed' flaw.
    """
    times, rms = frame_rms(y, sr, frame_s, hop_s)
    peaks = _envelope_peaks(times, rms)
    total_s = len(y) / sr
    out = []
    for s, e, c in _windows(total_s, win_s, win_hop_s):
        n = int(((peaks >= s) & (peaks < e)).sum())
        dur = e - s
        out.append({"start_s": s, "end_s": e, "center_s": c,
                    "rate_hz": n / dur if dur > 0 else 0.0, "n_peaks": n})
    return out


if __name__ == "__main__":
    # Self-check: a fast tone-burst train has a higher syllable rate than a slow one,
    # and a steady tone has lower local F0 std than a wobbling one.
    sr = 16000

    def _bursts(period_s, dur_s=3.0):
        t = np.arange(int(dur_s * sr)) / sr
        tone = 0.3 * np.sin(2 * np.pi * 150 * t)
        gate = (np.sin(2 * np.pi * (1.0 / period_s) * t) > 0).astype(float)
        return tone * gate

    slow = local_syllable_rate(_bursts(0.5), sr, 0.025, 0.010, 1.0, 0.5)
    fast = local_syllable_rate(_bursts(0.2), sr, 0.025, 0.010, 1.0, 0.5)
    assert max(w["rate_hz"] for w in fast) > max(w["rate_hz"] for w in slow)

    from .f0 import f0_contour
    t = np.arange(int(3.0 * sr)) / sr
    wob = 150 + 10 * np.sin(2 * np.pi * 2 * t)
    y_wob = 0.3 * np.sin(2 * np.pi * np.cumsum(wob) / sr)
    y_flat = 0.3 * np.sin(2 * np.pi * 150 * t)
    tw, fw = f0_contour(y_wob, sr, 65, 400, 0.010)
    tf, ff = f0_contour(y_flat, sr, 65, 400, 0.010)
    std_wob = min(w["std_hz"] for w in local_f0_std(tw, fw, 1.0, 0.5))
    std_flat = min(w["std_hz"] for w in local_f0_std(tf, ff, 1.0, 0.5))
    assert std_flat < std_wob
    print("local features OK")
