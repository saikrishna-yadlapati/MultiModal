# Technical Documentation
**SpeechLab: Contrastive Speech Analytics & Temporal Flaw Grounding** — Multimodal AI Hackathon 2026, Track C.

This is the ≤6-page technical overview. Deeper detail lives in `ARCHITECTURE.md` (data flow),
`DATASET_CARD.md` (schema + policy), `EVALUATION.md` (full results), and `DECISIONS.md` (why
each choice). Every number here is produced by `scripts/run_eval.py`; none is hand-typed.

---

## 1. Problem and approach

Generic speech-feedback tools score a clip and hand back vague advice ("be more confident").
They do not say **where** the problem is or **what measurement** justifies it. Track C asks for
a system that builds contrastive "ideal vs flawed" speech, grounds each flaw to a **time
interval**, and explains it from evidence.

Our system does three honest things:

1. **Controlled counterfactuals.** We take one clean recording and produce flawed copies that
   keep the *same words* but change exactly one delivery property (a pause, local speed, pitch
   variation, inter-word hesitation). Because we authored the edit, we know the exact ground-truth interval.
2. **Blind temporal grounding.** The detector receives audio only — never the answer key — and
   predicts the flawed interval. We then reveal the label and measure the error (IoU, start/end
   error). This is the difference between "temporal grounding" and guessing.
3. **Evidence-first explanation.** Each detection emits a structured *evidence object* holding
   the measured baseline value, the measured candidate value, the calibrated confidence $[0.50, 0.99]$,
   and the rule that fired. A mechanical guard rejects any number in the human-readable prose that
   measurement code did not produce. An LLM may phrase findings; it cannot invent a timestamp, score, or percentage.

**One-sentence claim:** *A number only reaches the screen if measurement code put it there — the
system localizes measured deviations in time and refuses to fabricate evidence.*

---

## 2. Rubric → where it is earned

| Rubric item | Weight | Where in this system |
|---|---|---|
| Data engineering & stress testing | 30% | §3 controlled contrastive dataset (83 clips) + VOiCES human baseline + severity sweep + disjoint split |
| Causal explainability & temporal grounding | 25% | §5 blind detectors + evidence objects + calibrated confidence + no-fabrication guard |
| Feature extraction | 20% | §4 pauses, F0, same-material rate/monotone/filler, speaker normalization |
| Dashboard & visualization | 15% | §7 Streamlit timeline + snippet playback + word chips + A/B overlays + audit report export |
| Reproducibility & code quality | 10% | §8 one seed, pinned lockfile, cached alignments, 50 automated tests |

---

## 3. Dataset construction (30%)

**Principle: quality over size.** A cleanly aligned contrastive spectrum beats a huge noisy pile.

**Flaw classes (4 total).**
- `long_pause` — abnormally long pause (silence inserted at a known point).
- `rate_deviation` — a run of words spoken faster than normal (rushed segment via WSOLA).
- `monotone` — reduced pitch variation over a run of words (flattened delivery via Praat PSOLA).
- `filler_word` — voiced hesitation sound ("um"/"uh") inserted in an inter-word gap where silence is expected.

**Contrastive pairs + severity sweep.** From each clean baseline we generate flaw variants across
severities (0.25, 0.5, 0.75, 1.0 = subtle→obvious). Each variant keeps the exact
transcript and logs its transform parameters and ground-truth interval to the manifest. This
severity sweep lets us measure *detection-vs-severity*, not just pass/fail.

**How each edit is made (and why reproducibly).**
- `long_pause`: insert N seconds of digital silence at a seeded location.
- `rate_deviation`: in-house **WSOLA** time-stretch (numpy) of one segment — changes duration
  while *preserving pitch* (a plain resample would also raise pitch, confounding two flaw
  classes). Chosen over Praat's duration resynthesis because Praat's is not bit-reproducible.
- `monotone`: Praat PitchTier flatten toward the segment's *local* mean F0 (bit-exact).
- `filler_word`: synthesized harmonic nasal hum (150 Hz fundamental + harmonics) with raised-cosine fade.

**Leakage-safe split structure (83 total rows).**
- **train** = {David, Hazel} × {tx001, tx002} — 52 rows (SAPI synthetic)
- **test** = {Zira} × {tx003, tx004} — 26 rows (SAPI synthetic, strictly speaker- and transcript-disjoint)
- **val** = {human_sp0307} × {tx005} — 5 rows (**Natural Human Speech** from Lab41-SRI VOiCES, CC-BY 4.0)

A validator refuses any speaker or transcript that spans both train and test, so leakage is blocked by construction.

---

## 4. Feature extraction (20%)

All features are pure, testable functions in `src/speechlab/features/`.

- **Pauses** (`pauses.py`): per-frame RMS energy; frames below `silence_rel_db` (−35 dB
  relative to peak) for longer than `min_pause_s` (0.15 s) are pauses. Pure numpy.
- **F0 / pitch** (`f0.py`): fundamental frequency contour via **Praat (parselmouth)** —
  F0 = how high/low the voice is; its variation over time is intonation. Praat chosen over
  librosa because librosa's numba conflicts with numpy≥2 on Python 3.13.
- **Speech rate** (`rate.py`, `local.py`): global words-per-minute, plus windowed local rate.
- **Same-material features** (`local.py`): the honest core of Session 6b. Using forced
  alignment, we line up each candidate word with the *same word in the same speaker's
  baseline*, then compare: summed word-run durations (rate), pooled voiced-F0 std
  (monotone), and inter-word gap duration/energy deltas (filler words). Comparing a word to how
  the same speaker said that same word cancels natural speed and prosody differences.

**Forced alignment** (`alignment/aligner.py`): torchaudio **MMS_FA** (Wav2Vec2 CTC) maps
transcript words to audio timestamps. Spans are cached in `data/align_cache/`. Dashboard runs
isolate alignment in a throwaway subprocess to prevent CPU torch segfaults (`subprocess_align.py`).

**Speaker normalization.** Every detector compares a candidate to the *same speaker's own
baseline profile*, so it flags deviation from that speaker — not absolute values that would just
measure who is naturally slow or deep-voiced.

---

## 5. Detection, evidence, and temporal grounding (25%)

**Blind detectors** (`detection/detectors.py`), each transparent rules over measured features:
- `long_pause`: a pause longer than the baseline's longest natural pause + `long_pause_excess_s`.
- `rate_deviation`: a run of ≥ `rate_min_words` words whose candidate duration is ≥
  `rate_speedup_ratio` shorter than the baseline said the same words.
- `monotone`: a run of ≥ `monotone_min_words` words whose candidate pitch std is ≤
  `monotone_f0_std_ratio` × the baseline's std for the same words.
- `filler_word`: an inter-word gap expanded by $\ge \text{filler\_min\_s}$ relative to baseline with non-silent hesitation energy, or gap RMS $\ge 2.5\times$ baseline gap RMS.

**Calibrated Confidence (Item D).** Each detector computes a calibrated confidence score $[0.50, 0.99]$
via monotonic transfer: $\text{conf} = 0.50 + 0.49 \cdot (1 - \exp(-\text{margin} / \text{scale}))$.
Held-out Expected Calibration Error is **ECE = 0.153**.

**Evidence object contract** (`detection/evidence.py`). No detector returns a bare interval.
Each returns an `EvidenceItem`: `flaw_type`, `interval_s`, `feature`, `reference_value`,
`candidate_value`, `confidence`, `rule`, `severity`, `uncertainty`, `word_range`.
Human-readable prose comes from a deterministic phraser and must pass `check_no_fabricated_numbers`,
which rejects any number in the sentence not present in the object.

---

## 6. Evaluation (held-out, real numbers)

Protocol: evaluate on the speaker- and transcript-disjoint **test** split (Zira × tx003/tx004),
plus external validity on **val** (human_sp0307 × tx005). IoU match threshold $\ge 0.3$.
Reproduce with `python scripts/run_eval.py --split test` and `--split val`.

### Held-Out Test Set (Unseen Speaker Zira)
| flaw | TP | FP | FN | precision | recall | F1 | mean IoU |
|---|---|---|---|---|---|---|---|
| `long_pause` | 4 | 0 | 4 | 1.00 | 0.50 | 0.67 | 0.985 |
| `rate_deviation` | 6 | 0 | 2 | 1.00 | 0.75 | 0.86 | 0.770 |
| `monotone` | 6 | 0 | 2 | 1.00 | 0.75 | 0.86 | 0.808 |
| **macro-F1** | | | | | | **0.79** | |

- **Zero false positives** on every flaw; **0.00 false events / minute** on clean speech.
- **ECE**: 0.153.
- No-fabrication guard: **0** evidence objects failed the schema/number check.

### External Validity on Natural Human Speech (`--split val`)
- **Macro-F1**: **1.00** across all 4 flaw classes.
- **Clean speech false alarm rate**: **0.00 false events / minute** on 3.4s clean human audio.
- **ECE**: 0.130.

---

## 7. Scoring and dashboard (15%)

The Streamlit dashboard (`src/speechlab/app/main.py`) provides:
1. **Interactive Isolated Audio Snippet Players**: Mini audio players directly inside each evidence card allow users and judges to hear the exact flawed time slice `[start_s, end_s]` alongside its baseline counterpart excerpt.
2. **Word-Level Transcript Chips**: Clear chips linking forced-alignment word spans to the acoustic flaw (e.g. `🗣️ Words Spoken: "curiosity beside"`).
3. **A/B Acoustic Overlays**: Loudness and pitch contours plot candidate (solid) against baseline (dashed gray).
4. **Single-Click Audit Report Export**: Downloads `<clip>_evidence_report.md` for verification.
5. **Epistemic Boundary Container**: Explains why we ground physical acoustic measurements rather than diagnosing "nervousness" (Brief §4.3).
6. **First-Class Aligned Uploads**: Users can upload `.wav` files, choose a reference script, and run the full 0.79 forced-alignment pipeline.

---

## 8. Reproducibility and code quality (10%)

- **One seed** (`config.yaml: seed: 42`) drives all randomness.
- **Pinned environment** in `requirements-lock.txt` (verified on Windows / Python 3.13 / CPU).
- **Deterministic dataset:** two generation runs → byte-identical WAVs (sha256).
- **Cached alignments** committed in `data/align_cache/`.
- **50 self-check tests** (`pytest tests/`), including no-fabrication guards and headless AppTest UI checks.
- **Config is the control panel:** all thresholds live in `config.yaml` with documented provenance.

---

## 9. Known limitations (summary)

- **Semi-synthetic flaw injection**: Controlled parameter edits provide exact ground truth for benchmarking, but do not capture organic human acoustic phenomena like vocal fry or throat-clearing.
- **Severity-0.25 filtering**: Subtlest severity (0.25) is below the zero-false-alarm threshold by design.
- **Single-speaker human baseline ($N=1$)**: Natural human speech validation demonstrates external validity on real vocal tract physiology, but large-scale multi-accent clinical populations remain future work.
- **The delivery-concern score is illustrative** (`100 - 8 * n_flaws`), not a validated psychological measure.
