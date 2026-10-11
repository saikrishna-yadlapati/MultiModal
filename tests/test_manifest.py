"""Self-checks for the dataset manifest schema, I/O, and leakage validator.

Pure stdlib, fast. Run: python -m pytest tests/test_manifest.py
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speechlab.dataset.manifest import (  # noqa: E402
    ManifestRow, make_pair_id, make_sample_id,
    read_manifest, validate_manifest, write_manifest,
)


def _pair(speaker="spk01", tx="tx01", split="train"):
    pid = make_pair_id(speaker, tx)
    base = ManifestRow(make_sample_id(pid, "baseline"), pid, speaker, tx,
                       "hello world", "data/raw/x.wav", 16000, "baseline",
                       "clean", split=split)
    var = ManifestRow(make_sample_id(pid, "variant", "long_pause", 1.0), pid,
                      speaker, tx, "hello world", "data/variants/x.wav", 16000,
                      "variant", "long_pause", severity=1.0, gt_start_s=1.0,
                      gt_end_s=2.5, split=split)
    return base, var


def test_valid_manifest_passes_and_roundtrips(tmp_path):
    rows = list(_pair())
    assert validate_manifest(rows) == []
    p = tmp_path / "m.csv"
    write_manifest(rows, p)
    assert read_manifest(p) == rows


def test_variant_without_ground_truth_fails():
    base, var = _pair()
    var.gt_start_s = None
    var.gt_end_s = None
    assert any("ground-truth" in p for p in validate_manifest([base, var]))


def test_speaker_leakage_across_splits_detected():
    # Same speaker spk01 appears in both train and test -> leakage.
    train_base, train_var = _pair(speaker="spk01", tx="tx01", split="train")
    test_base, test_var = _pair(speaker="spk01", tx="tx02", split="test")
    problems = validate_manifest([train_base, train_var, test_base, test_var])
    assert any("LEAKAGE" in p and "speaker_id" in p for p in problems)


def test_pair_split_inconsistency_detected():
    base, var = _pair(split="train")
    var.split = "test"  # variant disagrees with its baseline
    assert any("split-inconsistent" in p for p in validate_manifest([base, var]))


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_valid_manifest_passes_and_roundtrips(pathlib.Path(d))
    test_variant_without_ground_truth_fails()
    test_speaker_leakage_across_splits_detected()
    test_pair_split_inconsistency_detected()
    print("all manifest self-checks passed")
