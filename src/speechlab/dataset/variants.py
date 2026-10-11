"""Controlled, seeded flaw generators for the contrastive dataset (Session 3).

Three transforms, one per MVP flaw class (docs/DATASET_CARD.md):
  - long_pause     : insert a known silence  -> reuse dataset.inject.inject_pause
  - rate_deviation : speed up (rush) ONE segment, pitch preserved (WSOLA, pure numpy)
  - monotone       : flatten the pitch of ONE segment toward its own mean (Praat PSOLA)

Each generator returns (new_y, gt_interval, params):
  new_y        : the modified audio (float64, same sample rate)
  gt_interval  : (start_s, end_s) of the affected region IN THE OUTPUT audio — the
                 exact ground truth a blind detector is scored against.
  params       : the exact knobs used, logged to the manifest for reproducibility.

Reproducibility: all three are bit-exact across runs. Pause and rate are pure numpy;
pitch-flatten uses Praat's pitch-tier resynthesis, which is deterministic. (Praat's
DURATION resynthesis is NOT — that is why rate uses WSOLA instead of Praat. See
docs/DECISIONS.md.)
"""
from __future__ import annotations

import numpy as np
import parselmouth
from parselmouth.praat import call

from .inject import inject_pause  # noqa: F401  (re-exported for callers/tests)


def _wsola(x: np.ndarray, factor: float, frame: int, tol: int) -> np.ndarray:
    """WSOLA time-stretch of 1-D signal `x` by `factor` (output_len ~= len*factor).

    WSOLA = Waveform Similarity Overlap-Add: copy overlapping windowed frames from
    the input to the output at a different spacing, and before copying each frame
    nudge its read position by up to +/-`tol` samples to the spot that best lines up
    (max cross-correlation) with the waveform already laid down. That alignment is
    what keeps the PITCH intact while the DURATION changes — a plain resample would
    shift pitch too. Pure numpy => bit-exact reproducible (unlike Praat's resynth).
    """
    hs = frame // 2                       # synthesis hop (fixed output spacing)
    ha = max(1, int(round(hs / factor)))  # analysis hop (input spacing); <hs => stretch
    ov = frame - hs                       # overlap length between adjacent frames
    win = np.hanning(frame)
    out_len = int(np.ceil(len(x) * factor)) + frame
    out = np.zeros(out_len)
    norm = np.zeros(out_len)              # summed window weight, for overlap normalization
    a, s = 0, 0
    expect = None                         # waveform the next frame's head should resemble
    while a + frame <= len(x) and s + frame <= out_len:
        if expect is None:
            delta = 0
        else:
            lo = max(-tol, -a)
            hi = min(tol, len(x) - frame - a)
            best_d, best_c = 0, -np.inf
            for d in range(lo, hi + 1):
                head = x[a + d: a + d + ov]
                c = float(np.dot(head, expect))
                if c > best_c:
                    best_c, best_d = c, d
            delta = best_d
        seg = x[a + delta: a + delta + frame] * win
        out[s: s + frame] += seg
        norm[s: s + frame] += win
        # the input region that naturally follows this frame by one synthesis hop:
        nxt = a + delta + hs
        expect = x[nxt: nxt + ov]
        a += ha
        s += hs
    nz = norm > 1e-6
    out[nz] /= norm[nz]
    return out[: int(round(len(x) * factor))]


def change_rate(y: np.ndarray, sr: int, start_s: float, end_s: float, factor: float,
                frame_s: float = 0.04, tol_s: float = 0.01):
    """Time-stretch ONLY the segment [start_s, end_s] by `factor` (pitch preserved).

    Takes:
      y, sr        : audio and its sample rate.
      start_s/end_s: the segment of the ORIGINAL clip to re-time.
      factor       : output/input duration ratio for that segment. <1 = faster
                     (rushed), >1 = slower (dragged). 1.0 = no change.
      frame_s/tol_s: WSOLA window and search tolerance (seconds); defaults are fine.
    Returns:
      (new_y, (gt_start, gt_end), params). The segment's new length is
      (end_s-start_s)*factor, so in the output it spans [start_s, start_s+that].
      Audio before the segment is untouched; audio after it shifts by the delta.
    Flow: feeds make_variants, which writes new_y + the GT interval to the manifest.
    Knob: `variants.rate_deviation.factor_at_sev1` in config.yaml sets factor at severity 1.
    """
    i0 = max(0, int(round(start_s * sr)))
    i1 = min(len(y), int(round(end_s * sr)))
    frame = max(8, int(round(frame_s * sr)))
    tol = max(1, int(round(tol_s * sr)))
    stretched = _wsola(y[i0:i1].astype(np.float64), factor, frame, tol)
    new_y = np.concatenate([y[:i0], stretched, y[i1:]]).astype(np.float64)
    gt = (i0 / sr, i0 / sr + len(stretched) / sr)
    params = {"start_s": start_s, "end_s": end_s, "factor": factor,
              "frame_s": frame_s, "tol_s": tol_s, "method": "wsola"}
    return new_y, gt, params


def flatten_pitch(y: np.ndarray, sr: int, start_s: float, end_s: float,
                  k: float, fmin: float, fmax: float):
    """Flatten pitch variation in [start_s, end_s] toward the clip's mean F0.

    Takes:
      y, sr        : audio and sample rate.
      start_s/end_s: segment to flatten (duration is NOT changed).
      k            : variation kept. new_f0 = mean + k*(f0 - mean). k=0 => fully
                     monotone, k=1 => unchanged. (make_variants sets k = 1 - severity.)
      fmin/fmax    : pitch search range (config features.f0_min_hz / f0_max_hz).
    Returns:
      (new_y, (start_s, end_s), params). GT interval == the flattened segment
      (same length as input, pitch resynthesis preserves duration).
    Knob: severity -> k mapping lives in make_variants; fmin/fmax in config.features.
    """
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    man = call(snd, "To Manipulation", 0.01, fmin, fmax)
    pt = call(man, "Extract pitch tier")
    # LOCAL time-weighted mean over the segment (not the whole-clip point mean): forcing
    # pitch to a value far outside the segment's natural range makes PSOLA glitch
    # (octave errors). Flattening toward the pitch the speaker already used stays clean
    # and is the honest meaning of "monotone" — same phrase, pitch movement removed.
    mean = call(pt, "Get mean (curve)", start_s, end_s)
    if not np.isfinite(mean):
        raise ValueError(
            f"flatten_pitch: segment [{start_s},{end_s}] has no measurable pitch "
            "(mostly unvoiced); pick a voiced segment.")
    # Praat PitchTier Formula: `x` = a point's time, `self` = its frequency.
    # Only points inside the segment are pulled toward the mean; the rest keep `self`.
    call(pt, "Formula",
         f"if x >= {start_s} and x <= {end_s} then {mean} + {k}*(self-{mean}) else self fi")
    call([man, pt], "Replace pitch tier")
    out = call(man, "Get resynthesis (overlap-add)")
    new_y = np.asarray(out.values[0], dtype=np.float64)
    gt = (start_s, end_s)
    params = {"start_s": start_s, "end_s": end_s, "k": k, "mean_hz": float(mean),
              "method": "psola_pitch_flatten"}
    return new_y, gt, params


if __name__ == "__main__":
    # Self-check (no model download, pure Praat): a 150 Hz sine with wobble.
    sr = 16000
    t = np.arange(int(3.0 * sr)) / sr
    # frequency wobbles 140..160 Hz so there is pitch variation to flatten
    inst = 150 + 10 * np.sin(2 * np.pi * 2 * t)
    phase = 2 * np.pi * np.cumsum(inst) / sr
    y = 0.3 * np.sin(phase)

    ry, rgt, rp = change_rate(y, sr, 1.0, 2.0, 0.5)
    assert abs(rgt[1] - 1.5) < 0.05, rgt           # 1.0s segment -> 0.5s
    assert abs(len(ry) / sr - 2.5) < 0.1, len(ry)  # 3.0s clip loses 0.5s

    fy, fgt, fp = flatten_pitch(y, sr, 1.0, 2.0, 0.0, 65, 400)
    assert len(fy) == len(y), "flatten must preserve length"
    assert fgt == (1.0, 2.0)
    print("variants OK")
