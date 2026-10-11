"""Session 6a: the dataset plan builds a genuine speaker- AND transcript-disjoint
split, and such a manifest passes the leakage validator.

Pure logic only — imports the plan and the manifest schema, synthesizes no audio and
downloads no model, so it runs in milliseconds without SAPI.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from speechlab.dataset.manifest import (  # noqa: E402
    ManifestRow, make_pair_id, make_sample_id, validate_manifest,
)
from speechlab.dataset.plan import (  # noqa: E402
    TEST_SPEAKERS, TEST_TRANSCRIPTS, TRAIN_SPEAKERS, TRAIN_TRANSCRIPTS,
    TRANSCRIPTS, VOICES, planned_pairs,
)


def test_plan_is_disjoint_on_both_axes():
    assert set(TRAIN_SPEAKERS).isdisjoint(TEST_SPEAKERS)
    assert set(TRAIN_TRANSCRIPTS).isdisjoint(TEST_TRANSCRIPTS)


def test_plan_has_both_splits_and_known_voices():
    pairs = planned_pairs()
    splits = {s for _, _, s in pairs}
    assert splits == {"train", "test"}, splits
    assert sum(s == "train" for _, _, s in pairs) >= 1
    assert sum(s == "test" for _, _, s in pairs) >= 1
    # every planned pair names a real voice + a real transcript
    for spk, tx, _ in pairs:
        assert spk in VOICES and TRANSCRIPTS[tx]


def test_each_pair_split_agrees_with_both_sets():
    for spk, tx, split in planned_pairs():
        by_speaker = "train" if spk in TRAIN_SPEAKERS else "test"
        by_transcript = "train" if tx in TRAIN_TRANSCRIPTS else "test"
        assert split == by_speaker == by_transcript, (spk, tx, split)


def _rows_from_plan():
    """A manifest skeleton (baseline + one variant per pair) following the plan,
    with fabricated-but-valid ground truth — enough to exercise the leakage rules."""
    rows = []
    for spk, tx, split in planned_pairs():
        pid = make_pair_id(spk, tx)
        rows.append(ManifestRow(
            make_sample_id(pid, "baseline"), pid, spk, tx, TRANSCRIPTS[tx],
            f"data/raw/{pid}.wav", 22050, "baseline", "clean", split=split))
        rows.append(ManifestRow(
            make_sample_id(pid, "variant", "long_pause", 1.0), pid, spk, tx,
            TRANSCRIPTS[tx], f"data/variants/{pid}__lp.wav", 22050, "variant",
            "long_pause", severity=1.0, gt_start_s=5.0, gt_end_s=7.0, split=split))
    return rows


def test_planned_manifest_passes_leakage_validator():
    assert validate_manifest(_rows_from_plan()) == []


def test_swapping_a_test_speaker_into_train_is_caught():
    rows = _rows_from_plan()
    # Force one Zira (test) row into train -> speaker now spans two splits.
    for r in rows:
        if r.speaker_id in TEST_SPEAKERS:
            r.split = "train"
            break
    problems = validate_manifest(rows)
    assert any("LEAKAGE" in p for p in problems), problems
