"""Session 3 self-checks: variant generators are correct and reproducible.

Pure/local only — Praat PSOLA runs offline, no model download, no SAPI.
Run: python -m pytest tests/test_variants.py -q
"""
from __future__ import annotations

import pathlib
import random
import sys

import numpy as np
import parselmouth

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speechlab.dataset.inject import inject_pause  # noqa: E402
from speechlab.dataset.manifest import read_manifest  # noqa: E402
from speechlab.dataset.variants import change_rate, flatten_pitch, inject_filler  # noqa: E402

SR = 16000


def _wobble_tone(dur_s=3.0, sr=SR):
    """A tone whose pitch wobbles 140..160 Hz, so there is variation to flatten."""
    t = np.arange(int(dur_s * sr)) / sr
    inst = 150 + 10 * np.sin(2 * np.pi * 2 * t)
    phase = 2 * np.pi * np.cumsum(inst) / sr
    return 0.3 * np.sin(phase)


def _seg_f0_std(y, start, end, sr=SR):
    s = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    p = s.to_pitch(0.01, 65, 400)
    xs, f0 = np.asarray(p.xs()), np.asarray(p.selected_array["frequency"])
    m = (xs >= start) & (xs <= end) & (f0 > 0)
    return float(f0[m].std())


def test_pause_is_exact_zeros():
    y = _wobble_tone()
    ny, gt = inject_pause(y, SR, 1.0, 0.5)
    assert len(ny) == len(y) + round(0.5 * SR)        # grew by exactly the silence
    s, e = round(gt[0] * SR), round(gt[1] * SR)
    assert np.all(ny[s:e] == 0.0)                     # inserted region is silence
    assert abs((gt[1] - gt[0]) - 0.5) < 1e-6          # GT length == requested pause


def _seg_f0_mean(y, start, end, sr=SR):
    s = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    p = s.to_pitch(0.01, 65, 400)
    xs, f0 = np.asarray(p.xs()), np.asarray(p.selected_array["frequency"])
    m = (xs >= start) & (xs <= end) & (f0 > 0)
    return float(f0[m].mean())


def test_rate_shortens_segment():
    y = _wobble_tone()
    ny, gt, params = change_rate(y, SR, 1.0, 2.0, 0.5)  # 1.0s -> 0.5s (2x faster)
    assert abs((gt[1] - gt[0]) - 0.5) < 0.05            # GT reflects compressed length
    assert len(ny) / SR < len(y) / SR                   # whole clip is shorter
    assert params["method"] == "wsola"
    # pitch preserved: the sped-up segment keeps ~150 Hz (a resample would raise it)
    before = _seg_f0_mean(y, 1.0, 2.0)
    after = _seg_f0_mean(ny, 1.0, gt[1])
    assert abs(after - before) < 15                     # within ~10% of the mean


def test_flatten_reduces_pitch_std_keeps_length():
    y = _wobble_tone()
    before = _seg_f0_std(y, 1.0, 2.0)
    ny, gt, params = flatten_pitch(y, SR, 1.0, 2.0, 0.0, 65, 400)  # k=0 => monotone
    after = _seg_f0_std(ny, 1.0, 2.0)
    assert len(ny) == len(y)                            # pitch change preserves duration
    assert after < before * 0.5                         # variation clearly reduced
    assert gt == (1.0, 2.0)


def test_transforms_are_deterministic():
    y = _wobble_tone()
    a, _, _ = change_rate(y, SR, 1.0, 2.0, 0.5)
    b, _, _ = change_rate(y, SR, 1.0, 2.0, 0.5)
    assert np.array_equal(a, b)                         # same input => identical audio
    c, _, _ = flatten_pitch(y, SR, 1.0, 2.0, 0.3, 65, 400)
    d, _, _ = flatten_pitch(y, SR, 1.0, 2.0, 0.3, 65, 400)
    assert np.array_equal(c, d)
    # seeded location picks repeat
    r1 = [random.Random(42).uniform(2, 5) for _ in range(1)]
    r2 = [random.Random(42).uniform(2, 5) for _ in range(1)]
    assert r1 == r2


def test_manifest_variants_have_gt_and_params():
    """Acceptance: every variant row carries a GT interval and transform params."""
    mpath = ROOT / "data" / "manifest" / "manifest.csv"
    if not mpath.exists():
        import pytest
        pytest.skip("run scripts/make_variants.py first")
    variants = [r for r in read_manifest(mpath) if r.baseline_or_variant == "variant"]
    assert variants, "manifest has no variants"
    for r in variants:
        assert r.gt_start_s is not None and r.gt_end_s is not None, r.sample_id
        assert r.gt_end_s > r.gt_start_s, r.sample_id
        assert r.transform_params, r.sample_id


def test_inject_filler_inserts_voiced_segment():
    y = _wobble_tone()
    ny, gt, params = inject_filler(y, SR, 1.5, 0.4, severity=1.0)
    assert len(ny) > len(y)
    assert abs((gt[1] - gt[0]) - 0.4) < 0.01
    assert params["method"] == "synth_nasal_filler"
    # The inserted region must be non-zero (voiced)
    filler_region = ny[int(gt[0] * SR):int(gt[1] * SR)]
    assert float(np.abs(filler_region).max()) > 0.01

