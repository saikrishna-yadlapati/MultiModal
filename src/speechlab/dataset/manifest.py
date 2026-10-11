"""Dataset manifest: schema, stable IDs, CSV I/O, and a leakage validator.

Schema is defined in docs/DATASET_CARD.md (one row per audio file). Deliberately
stdlib-only (csv + json + dataclasses) — no pandas until a real need appears (YAGNI).

The validator is the scientifically important part: it refuses a manifest where the
same speaker or transcript spans two splits (train/test leakage, context §7), where a
variant is missing its ground-truth interval, or where a baseline/variant pair disagrees
on its split. A silent bad split would quietly inflate every later metric.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

# Allowed enum values — kept here so the validator and any generator agree.
FLAW_TYPES = {"clean", "long_pause", "rate_deviation", "monotone"}
ROLES = {"baseline", "variant"}
SPLITS = {"train", "val", "test"}

# Fields serialized as JSON in the CSV (they hold a list/dict, not a scalar).
_JSON_FIELDS = {"word_range", "transform_params"}
# Fields parsed back to float when reading (empty string -> None).
_FLOAT_FIELDS = {"severity", "gt_start_s", "gt_end_s"}


@dataclass
class ManifestRow:
    """One audio file. Field order == docs/DATASET_CARD.md == CSV column order."""
    sample_id: str
    pair_id: str
    speaker_id: str
    transcript_id: str
    transcript_text: str
    audio_path: str
    sample_rate: int
    baseline_or_variant: str
    flaw_type: str
    severity: float | None = None
    gt_start_s: float | None = None
    gt_end_s: float | None = None
    word_range: list[int] | None = None
    transform_method: str = ""
    transform_params: dict | None = None
    alignment_method: str = ""
    alignment_status: str = ""
    source_url: str = ""
    license: str = ""
    split: str = "train"
    notes: str = ""


def make_pair_id(speaker_id: str, transcript_id: str) -> str:
    """Deterministic id linking a baseline to its variants (same speaker+transcript)."""
    return f"{speaker_id}__{transcript_id}"


def make_sample_id(pair_id: str, role: str, flaw_type: str = "clean",
                   severity: float | None = None) -> str:
    """Deterministic unique id for one audio file within a pair."""
    if role == "baseline":
        return f"{pair_id}__baseline"
    sev = "" if severity is None else f"_s{severity:g}"
    return f"{pair_id}__{flaw_type}{sev}"


# --- CSV I/O --------------------------------------------------------------

def _field_names() -> list[str]:
    return [f.name for f in fields(ManifestRow)]


def write_manifest(rows: list[ManifestRow], path: str | Path) -> None:
    """Write rows to CSV. JSON-encodes list/dict fields; None -> empty cell."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = _field_names()
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            d = asdict(r)
            for k, v in d.items():
                if v is None:
                    d[k] = ""
                elif k in _JSON_FIELDS:
                    d[k] = json.dumps(v)
            w.writerow(d)


def read_manifest(path: str | Path) -> list[ManifestRow]:
    """Read CSV back into ManifestRow objects, decoding JSON/numeric fields."""
    out: list[ManifestRow] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            d: dict = dict(raw)
            d["sample_rate"] = int(d["sample_rate"])
            for k in _FLOAT_FIELDS:
                d[k] = None if d[k] == "" else float(d[k])
            for k in _JSON_FIELDS:
                d[k] = None if d[k] == "" else json.loads(d[k])
            out.append(ManifestRow(**d))
    return out


# --- validation -----------------------------------------------------------

def validate_manifest(rows: list[ManifestRow]) -> list[str]:
    """Return a list of problems (empty == valid). Does not raise.

    Checks: unique ids, enum membership, variant ground-truth presence, and the
    two leakage rules (no speaker/transcript across splits; a pair shares one split).
    """
    problems: list[str] = []

    seen_ids: set[str] = set()
    for r in rows:
        if not r.sample_id or r.sample_id in seen_ids:
            problems.append(f"duplicate/empty sample_id: {r.sample_id!r}")
        seen_ids.add(r.sample_id)
        if r.baseline_or_variant not in ROLES:
            problems.append(f"{r.sample_id}: bad role {r.baseline_or_variant!r}")
        if r.flaw_type not in FLAW_TYPES:
            problems.append(f"{r.sample_id}: bad flaw_type {r.flaw_type!r}")
        if r.split not in SPLITS:
            problems.append(f"{r.sample_id}: bad split {r.split!r}")
        if r.baseline_or_variant == "baseline" and r.flaw_type != "clean":
            problems.append(f"{r.sample_id}: baseline must be flaw_type 'clean'")
        if r.baseline_or_variant == "variant":
            if r.flaw_type == "clean":
                problems.append(f"{r.sample_id}: variant must have a non-clean flaw_type")
            if r.gt_start_s is None or r.gt_end_s is None:
                problems.append(f"{r.sample_id}: variant missing ground-truth interval")
            elif r.gt_end_s <= r.gt_start_s:
                problems.append(f"{r.sample_id}: gt_end <= gt_start")

    # Leakage rule 1: a speaker or transcript must live in exactly one split.
    for key in ("speaker_id", "transcript_id"):
        split_map: dict[str, set[str]] = {}
        for r in rows:
            split_map.setdefault(getattr(r, key), set()).add(r.split)
        for val, splits in split_map.items():
            if len(splits) > 1:
                problems.append(
                    f"LEAKAGE: {key}={val!r} spans splits {sorted(splits)}")

    # Leakage rule 2: every row in a pair_id shares one split (variants inherit baseline's).
    pair_map: dict[str, set[str]] = {}
    for r in rows:
        pair_map.setdefault(r.pair_id, set()).add(r.split)
    for pid, splits in pair_map.items():
        if len(splits) > 1:
            problems.append(f"pair {pid!r} split-inconsistent: {sorted(splits)}")

    return problems


if __name__ == "__main__":
    # Self-check: a tiny valid manifest round-trips and validates clean.
    import tempfile

    pid = make_pair_id("spk01", "tx01")
    rows = [
        ManifestRow(make_sample_id(pid, "baseline"), pid, "spk01", "tx01",
                    "hello world", "data/raw/x.wav", 16000, "baseline", "clean",
                    split="train"),
        ManifestRow(make_sample_id(pid, "variant", "long_pause", 1.0), pid, "spk01",
                    "tx01", "hello world", "data/variants/x__pause.wav", 16000,
                    "variant", "long_pause", severity=1.0, gt_start_s=1.0,
                    gt_end_s=2.5, split="train"),
    ]
    assert validate_manifest(rows) == [], validate_manifest(rows)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.csv"
        write_manifest(rows, p)
        back = read_manifest(p)
    assert back == rows, "round-trip changed the rows"
    print("manifest OK")
