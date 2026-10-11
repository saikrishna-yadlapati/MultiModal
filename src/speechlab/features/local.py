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
