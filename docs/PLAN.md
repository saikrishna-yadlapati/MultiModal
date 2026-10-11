# PLAN — Session & milestone roadmap

Source of truth for scope is `HACKATHON_CONTEXT.md`. This file is the *execution* plan:
what each session delivers, its acceptance test, and the commit it earns.

Rule: a session is only "done" when its acceptance test passes on real data and the
result is recorded in `SESSION_LOG.md`. No fabricated results. Untested = marked untested.

## Rubric → where it is earned
| Rubric item | Weight | Sessions |
|---|---|---|
| Data engineering & stress testing | 30% | 2, 3 |
| Causal explainability & temporal grounding | 25% | 1, 4 |
| Feature extraction | 20% | 1, 4 |
| Dashboard & visualization | 15% | 5 |
| Reproducibility & code quality | 10% | 0, 6, 7 |

## Sessions

### Session 0 — Scaffold (this session)
- Repo structure, planning docs, memory, dependency list, stack decision (Streamlit).
- Acceptance: tree exists, docs readable, `pip install -r requirements.txt` plan documented.
- Commit: `chore: scaffold repo structure and planning docs`

### Session 1 — One-sample vertical slice
- Load one short WAV + transcript. Extract pauses, speech-rate, F0. Print a report.
- Inject one known flaw (e.g. a 1.5s pause at a known time) into a copy.
- Run the detector blind; verify it returns a timestamp near ground truth.
- Acceptance: `python scripts/run_slice.py <wav> <transcript>` prints features AND the
  injected-flaw detection with predicted interval vs ground truth. One self-check test passes.
- Commit: `feat: end-to-end single-sample feature + blind flaw detection`

### Session 2 — Dataset manifest + alignment decision
- Define manifest schema (see `DATASET_CARD.md`), stable IDs, CSV/JSON writer.
- Choose + test ONE forced aligner on a single sample (candidates in `ARCHITECTURE.md`).
  Document fallback if it fails in this environment. Do NOT install a heavy aligner before testing.
- Acceptance: manifest validates; alignment produces word timestamps on one sample OR
  fallback is documented with evidence of the failure.
- Commit: `feat: dataset manifest schema and forced-alignment integration`

### Session 3 — Controlled variants + severity sweep
- Deterministic (seeded) generators: pause-insert, local rate-change, pitch-flatten.
  Each logs transformation params and writes ground-truth intervals to the manifest.
- Severity sweep: produce a gradient per flaw type.
- Acceptance: generators are reproducible (same seed → same output); manifest has
  ground-truth intervals + params for every variant.
- Commit: `feat: controlled contrastive variant generators with severity sweep`

### Session 4 — Baseline detectors + evidence objects
- Transparent rule/threshold detectors with speaker-aware normalization.
- Strict evidence schema (feature, reference value, candidate value, interval, rule, uncertainty).
- Optional: LLM phrases the evidence object into prose (never invents numbers).
- Acceptance: detector outputs validated evidence objects; prose (if used) contains no
  number absent from the object.
- Commit: `feat: baseline detectors and structured evidence objects`

### Session 5 — Streamlit dashboard
- Upload/select candidate + baseline. Playback. Synced waveform/feature timeline.
  Highlighted intervals. Evidence panel. Rubric score. Limitations notice.
- Acceptance: `streamlit run src/speechlab/app/main.py` loads, analyzes a sample, shows intervals.
- Commit: `feat: interactive Streamlit dashboard`

### Session 6 — Held-out evaluation (split into 6a + 6b on 2026-10-11)
The original single-session plan assumed a dataset to evaluate on. Sessions 1–5 built it on
ONE speaker / ONE transcript (all `train`), so a real speaker/transcript-disjoint held-out
split did not yet exist. Owner approved splitting Session 6:

#### Session 6a — Expand dataset for a real held-out split ✅ done
- 3 SAPI voices × 4 transcripts; generate only the consistent grid blocks so the split is
  speaker- AND transcript-disjoint (train = David/Hazel × tx001/tx002; test = Zira × tx003/tx004).
- Split policy in `src/speechlab/dataset/plan.py`; built by `scripts/make_dataset.py`
  (supersedes `make_variants.py`). Validator must stay clean; WAVs reproducible (sha256).
- Acceptance: `python scripts/make_dataset.py` writes a validator-clean manifest with both
  splits; split test asserts disjointness on both axes; two runs → byte-identical WAVs.
- Commit: `feat: multi-speaker/transcript dataset with held-out split`

#### Session 6b — Fixes + metrics (next)
- Fix monotone (word-aligned same-material comparison) and rate_deviation (alignment-based
  local rate) using the Session 2 aligner. Tune thresholds on the TRAIN split only.
- Event precision/recall/F1 by flaw class, localization error (IoU / start-end error),
  false-positive rate on clean speech, detection-vs-severity — all on the held-out TEST set.
- Include failure cases. Real numbers only.
- Acceptance: `python scripts/run_eval.py` prints metrics from the held-out set; results in `EVALUATION.md`.
- Commit: `feat: held-out evaluation harness with real metrics`

### Session 7 — Submission polish
- README (prereqs, setup, run), docs ≤6 pages, AI-tool disclosure, dataset artifacts/licenses,
  demo storyboard. Fresh-environment reproduce check.
- Commit: `docs: submission-ready README, docs, and AI disclosure`

## Commit philosophy
A commit = one coherent, working, reviewable unit (a session or a complete sub-feature) —
not every file save. Suggested message given at each session end; you commit manually.
