# ARCHITECTURE

Working brand: TBD (do NOT use "SonicTrace" — name conflict noted in context §5).
Python package name (internal, placeholder): `speechlab`.

## Principle
Numbers come from code. The LLM may only *phrase* a validated evidence object — it never
produces a timestamp, measurement, score, or label. Keep modules minimal and swappable.
Prove the core on one sample before adding any heavy dependency.

## Data flow
```
baseline audio + transcript
        │
        ▼
   [alignment]  ── word/phrase timestamps
        │
        ▼
   [features]   ── per-window F0, rate, pauses, energy, MFCC (speaker-normalized)
        │
        ├──────────────► reference feature profile
        │
candidate audio ─► [features] ─► candidate profile
        │
        ▼
   [detection]  ── compare vs reference/rubric → flaw intervals (BLIND to ground truth)
        │
        ▼
 [evidence object]  ── strict schema: feature, ref value, cand value, interval, rule, uncertainty
        │
        ├──► [LLM phrasing]  (optional, prose only, no new numbers)
        │
        ▼
   [dashboard / report]   ── timelines, highlighted intervals, evidence, score, limits
        │
   [evaluation]  ── predicted intervals vs ground-truth labels → real metrics
```

## Modules (`src/speechlab/`)
- `config.py` — paths, seeds, feature params, detection thresholds (loaded from `config.yaml`).
- `features/` — acoustic feature extraction (librosa, parselmouth). Pure functions, testable.
- `alignment/` — forced alignment + fallback. Decision in Session 2.
- `dataset/` — manifest schema, IDs, variant generators, provenance/license tracking.
- `detection/` — transparent rule/threshold detectors, speaker normalization, evidence objects.
- `evaluation/` — splits, metrics (P/R/F1, IoU, localization error, FP rate, severity curve).
- `app/` — Streamlit dashboard (`main.py`).

Entry points live in `scripts/`; `src/speechlab` holds importable, tested logic.

## Forced-alignment candidates (decide + TEST in Session 2, pick ONE)
Trade-offs to resolve against this environment (Windows, CPU, reproducibility, time):
- **torchaudio Wav2Vec2 CTC forced alignment** — pip-installable, CPU-ok, heavier (torch).
- **WhisperX** — accurate word timestamps, pulls torch + whisper, heavier install.
- **aeneas** — lightweight, needs text, older but simple; good fallback.
- **Montreal Forced Aligner (MFA)** — strong but conda/kaldi, heavy on Windows.
Do not install any of these in Session 0. Test one on a single sample first; document failures.

## Evidence object contract (finalized in Session 4)
```
EvidenceItem:
  flaw_type: str              # e.g. "long_pause"
  interval_s: [start, end]    # from code
  feature: str                # e.g. "pause_duration"
  reference_value: float      # measured on baseline
  candidate_value: float      # measured on candidate
  rule: str                   # the threshold/comparison that fired
  severity: float | None
  uncertainty: str            # why confidence is limited, if so
  word_range: [i, j] | None
```
Rule for the LLM: every number in the prose MUST exist in the object. Enforce with a check.
