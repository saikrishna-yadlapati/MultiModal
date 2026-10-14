"""Structured evidence objects + the no-fabrication guard (Session 4).

The scientific contract of this project (docs/ARCHITECTURE.md): a NUMBER may only
come from measurement code, never from prose. An `EvidenceItem` is the handoff
record a detector produces: it says WHAT fired, WHERE (interval), the measured
reference-vs-candidate values, and the exact RULE that fired. Any human-readable
sentence must be derived from this object and may not contain a number the object
does not hold — `check_no_fabricated_numbers` enforces that mechanically, so an LLM
could later replace the template phraser without being able to smuggle in invented
timestamps, scores, or percentages.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class EvidenceItem:
    """One detected deviation. Field order == docs/ARCHITECTURE.md evidence contract."""
    flaw_type: str                       # "long_pause" / "rate_deviation" / "monotone" / "filler_word"
    interval_s: list[float]              # [start, end] in seconds — from code, not prose
    feature: str                         # measured quantity, e.g. "pause_duration_s"
    reference_value: float               # measured on the baseline (same speaker)
    candidate_value: float               # measured on the candidate
    rule: str                            # the comparison that fired (human-readable)
    severity: float | None = None        # 0..1 if the rule yields one, else None
    confidence: float | None = None      # 0..1 calibrated confidence score (Item D)
    uncertainty: str = ""                # why confidence is limited (always honest)
    word_range: list[int] | None = None  # affected word indices, when alignment is used


def compute_confidence(margin: float, scale: float) -> float:
    """Defensible acoustic confidence: saturating transfer from margin above threshold.

    Returns a score in [0.50, 0.99] rounded to 2 decimals.
    At margin=0 (boundary), returns 0.50 (minimum detection confidence).
    As margin grows relative to scale, confidence asymptotically approaches 1.0.
    """
    if scale <= 0:
        return 0.50
    delta = max(0.0, float(margin)) / float(scale)
    conf = 1.0 - 0.50 / (1.0 + delta)
    return round(min(0.99, max(0.50, conf)), 2)


def _fmt(x: float) -> str:
    """Single canonical number format shared by the phraser and the guard."""
    return f"{x:.2f}"


def validate_evidence(e: EvidenceItem) -> list[str]:
    """Return a list of structural problems (empty == valid). Does not raise."""
    p: list[str] = []
    if e.flaw_type not in {"long_pause", "rate_deviation", "monotone", "filler_word"}:
        p.append(f"bad flaw_type {e.flaw_type!r}")
    if (not isinstance(e.interval_s, (list, tuple)) or len(e.interval_s) != 2):
        p.append("interval_s must be [start, end]")
    elif e.interval_s[1] <= e.interval_s[0]:
        p.append("interval end <= start")
    elif e.interval_s[0] < 0:
        p.append("interval start < 0")
    for name in ("reference_value", "candidate_value"):
        v = getattr(e, name)
        if not isinstance(v, (int, float)):
            p.append(f"{name} is not numeric")
    if not e.rule:
        p.append("rule is empty")
    if not e.uncertainty:
        p.append("uncertainty is empty (must state why confidence is limited)")
    if e.severity is not None and not (0.0 <= e.severity <= 1.0):
        p.append("severity out of [0,1]")
    if e.confidence is not None and not (0.0 <= e.confidence <= 1.0):
        p.append("confidence out of [0,1]")
    return p


# Numbers not preceded by a letter (so the '0' in "F0" or "MFCC13" is ignored).
_NUM = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?")


def _allowed_numbers(e: EvidenceItem) -> set[str]:
    """Every number the prose is permitted to contain, in canonical format.

    Includes the numeric fields AND any number inside the code-authored `rule` string
    (thresholds like 0.30s are set by the detector, not invented by the phraser).
    """
    vals = [e.interval_s[0], e.interval_s[1], e.reference_value, e.candidate_value]
    if e.severity is not None:
        vals.append(e.severity)
    if e.confidence is not None:
        vals.append(e.confidence)
    if e.word_range is not None:
        vals.extend(e.word_range)
    out = set()
    for v in vals:
        out.add(_fmt(float(v)))
        out.add(f"{float(v):g}")      # also allow a trailing-zero-free form
    out.update(_NUM.findall(e.rule))  # thresholds etc. come from measurement code
    return out


def check_no_fabricated_numbers(text: str, e: EvidenceItem) -> list[str]:
    """Return numbers in `text` that are NOT backed by the evidence object.

    This is the guardrail: run it on any generated sentence (template or LLM). A
    non-empty result means the prose invented a number and must be rejected.
    """
    allowed = _allowed_numbers(e)
    return [tok for tok in _NUM.findall(text) if tok not in allowed]


def phrase_evidence(e: EvidenceItem) -> str:
    """Deterministic, number-safe sentence built ONLY from the object's fields.

    No LLM: every number comes straight from a field via `_fmt`, so the output
    passes `check_no_fabricated_numbers` by construction. An LLM phraser could slot
    in here later behind the same guard.
    """
    label = e.flaw_type.replace("_", " ")
    s, en = _fmt(e.interval_s[0]), _fmt(e.interval_s[1])
    cand, ref = _fmt(e.candidate_value), _fmt(e.reference_value)
    sev = "" if e.severity is None else f" (severity {_fmt(e.severity)})"
    conf = "" if e.confidence is None else f" [confidence {_fmt(e.confidence)}]"
    return (f"Possible {label} from {s}s to {en}s{sev}{conf}: {e.feature} measured {cand} "
            f"versus a baseline of {ref} for this speaker ({e.rule}). {e.uncertainty}")


if __name__ == "__main__":
    # Self-check: a valid item phrases cleanly and passes the guard; a sentence with
    # an invented number is caught.
    ev = EvidenceItem(
        flaw_type="long_pause", interval_s=[4.50, 6.00], feature="pause_duration_s",
        reference_value=0.90, candidate_value=1.50, severity=0.75,
        rule="pause exceeds baseline's longest pause (0.90s) by >= 0.30s",
        uncertainty="Pause length alone; no alignment, a shifted natural pause could overlap.")
    assert validate_evidence(ev) == [], validate_evidence(ev)
    good = phrase_evidence(ev)
    assert check_no_fabricated_numbers(good, ev) == [], good
    bad = good + " The speaker was 95 percent nervous."
    assert check_no_fabricated_numbers(bad, ev) == ["95"], check_no_fabricated_numbers(bad, ev)
    assert validate_evidence(EvidenceItem("monotone", [2.0, 1.0], "f", 1.0, 0.0,
                                          "r", uncertainty="u")) != []
    print("evidence OK")
