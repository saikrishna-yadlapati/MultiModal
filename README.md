# SpeechLab: Contrastive Speech Analytics & Temporal Flaw Grounding
_Multimodal AI Hackathon 2026 — Track C Submission._

A speech-delivery debugging system. It builds controlled contrastive pairs of
"ideal" vs "flawed" speech, locates each flaw **in time, blind** (the detector never sees the
answer key), explains it from **measured acoustic evidence** (not LLM guesses), and reports
**honest held-out metrics** on an unseen speaker speaking unseen sentences, plus real-world external validity on studio-recorded human speech.

> **Status:** End-to-end verified pipeline. Full test suite: **50 passed**. Held-out evaluation + natural human speech validation complete. See `docs/SESSION_LOG.md` for the full trajectory.

---

## Why It's Not Just "Ask an LLM"
- **Same transcript**, controlled delivery edits with logged parameters → **known ground-truth intervals**. We know exactly where the flaw is, how it was made, and its acoustic magnitude.
- The detector sees **audio only**, predicts the flawed interval, then reveals ground truth to measure error. That is rigorous temporal grounding, not uncalibrated guessing.
- Every explanation is built from a structured **evidence object** (measured baseline value, measured candidate value, calibrated confidence $[0.50, 0.99]$, the rule that fired).
- A mechanical guard (`check_no_fabricated_numbers`) rejects any number in prose that measurement code did not produce — so an LLM can phrase findings but **cannot invent a timestamp, score, or percentage**.

---

## Headline Results

### 1. Held-Out Benchmark (Test Split: Unseen Speaker + Unseen Transcripts)
Test = speaker `zira` on transcripts `tx003`/`tx004` (strictly speaker- and transcript-disjoint from train). Thresholds tuned on the TRAIN split only. IoU match threshold 0.3.

| Flaw Class | Precision | Recall | F1 | Mean IoU |
|---|---|---|---|---|
| `long_pause` | 1.00 | 0.50 | 0.67 | 0.985 |
| `rate_deviation` | 1.00 | 0.75 | 0.86 | 0.770 |
| `monotone` | 1.00 | 0.75 | 0.86 | 0.808 |
| **Macro-F1** | | | **0.79** | |

- **Zero false positives** on every flaw; **0.00 false events / minute** on clean speech.
- **Expected Calibration Error (ECE)**: **0.153** across held-out predictions (well-calibrated uncertainty).
- **Honesty guard**: **0** evidence objects failed the number check.
- **Honest miss**: Subtlest severity (0.25) is below the zero-false-alarm threshold — reported transparently.

### 2. External Validity on Natural Human Speech (`--split val`)
Ingested clean studio recording from the **Lab41-SRI VOiCES Corpus** (Speaker `sp0307`, Utterance `ch127535-sg0042`, CC-BY 4.0 license, 16 kHz mono).
- **Macro-F1 on human speech**: **1.00** across all 4 flaw classes (`long_pause`, `rate_deviation`, `monotone`, `filler_word`).
- **Clean human speech false alarms**: **0.00 false events / minute** (0 false alarms over clean human speech).
- **Calibration on human speech**: ECE = 0.130 with zero guard violations.

---

## Interactive Dashboard Features (`src/speechlab/app/main.py`)
- **A/B Acoustic Overlays**: Loudness timeline and pitch contour charts overlay the speaker's baseline reference (dashed gray) against the candidate signal (solid).
- **Isolated Flaw Audio Snippet Player**: Dedicated mini-players inside each evidence card allow users and judges to click and hear the exact flawed time slice `[start_s, end_s]` side-by-side with its baseline counterpart excerpt.
- **Word-Level Grounding Chips**: Displays the exact words spoken during the flaw (e.g. `🗣️ Words Spoken: "curiosity beside"`) or inter-word gap context.
- **First-Class Aligned Uploads**: Users can upload any `.wav` recording, select a practice target script, and run the full 0.79 forced-alignment pipeline.
- **Audit Report Export**: Single-click `st.download_button` exports a verifiable Markdown audit report (`<clip>_evidence_report.md`).
- **Epistemic Honesty Panel**: Explains why we measure physical acoustic deviations rather than ungrounded emotional states like "nervousness" (Brief §4.3).

---

## Project Layout
```
docs/              planning, architecture, dataset card, evaluation, decisions, session log
src/speechlab/     core package:
  features/        pauses, F0 (Praat), speech rate, windowed + same-material features
  alignment/       torchaudio MMS_FA forced aligner, subprocess isolation, span caching
  dataset/         manifest schema, split plan, controlled variant generators
  detection/       speaker-normalized blind detectors + evidence objects + honesty guard
  evaluation/      interval IoU / errors, held-out metrics, calibration ECE
  app/             Streamlit dashboard (pure analysis.py + interactive main.py)
scripts/           command-line entry points:
  run_eval.py      held-out evaluation harness (--split test / val / all)
  make_dataset.py  dataset generation (78 synthetic clips)
  ingest_human_baseline.py  VOiCES CC-BY 4.0 human speech baseline & variants
data/              raw baselines, variants, manifest.csv, pre-cached alignment spans
tests/             50 self-checks across logic, metrics, alignment, and dashboard UI
```

---

## Setup (Windows / macOS / Linux / CPU — Verified)
```bash
# 1. Create and activate virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
pip install -e . --no-deps
```

Exact pinned versions are frozen in `requirements-lock.txt`.
Cached alignment spans are pre-committed in `data/align_cache/`, allowing `run_eval.py` to run out-of-the-box without requiring an initial acoustic model download.

---

## Run Commands
```bash
# Run full unit test suite (50 passed)
pytest tests/ -q

# Run held-out evaluation on test split (Macro-F1 0.79, 0 FA/min)
python scripts/run_eval.py --split test

# Run external-validity evaluation on real human speech (Macro-F1 1.00)
python scripts/run_eval.py --split val

# Launch interactive Streamlit dashboard
streamlit run src/speechlab/app/main.py
```

---

## Dataset & Licensing
- **Corpus Size**: 83 samples across 4 speakers and 5 distinct transcripts.
- **Synthetic Speech**: Generated locally via Windows SAPI TTS (voices `David`, `Hazel`, `Zira`).
- **Human Speech**: Utterance `ch127535-sg0042` from **Lab41-SRI VOiCES Corpus** under Creative Commons Attribution 4.0 International (CC-BY 4.0).
- **Leakage Prevention**: Strictly assigned by speaker and transcript disjointness (`train`, `test`, `val`). Formally validated by `tests/test_dataset_split.py`.
- Full policy, provenance, and data cards in `docs/DATASET_CARD.md`.

---

## Honest Limitations
- **Semi-synthetic flaw generation**: Flaws on both synthetic and human voices are mathematically injected. This provides precise ground truth for algorithm benchmarking, but does not simulate all spontaneous speech phenomena (e.g. vocal fry or stammering repetitions).
- **Severity-0.25 filtering**: Subtle 0.25 severity flaws fall below detection cutoffs by design to maintain a zero false-alarm rate on clean speech.
- **Delivery-concern score**: The 0–100 delivery-concern score is an illustrative heuristic formula (`100 - 8 * n_flaws`), not a validated clinical or psychological metric.

---

## AI-Tool Disclosure
Built with the assistance of AI coding tools (Claude Code & Google Antigravity) for code generation, test scaffolding, and refactoring. **All acoustic calculations, timestamps, confidence scores, and metrics are produced deterministically by the underlying DSP and alignment code, verified by automated tests, and protected by the no-fabrication guard in `src/speechlab/detection/evidence.py`.**
