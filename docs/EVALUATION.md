# EVALUATION

Status: protocol defined (Session 0). No results yet. Never fabricate metrics —
every number here must come from `scripts/run_eval.py` on a held-out set.

## Protocol
- Split by speaker AND transcript (see `DATASET_CARD.md`). Evaluate on unseen samples only,
  never the files used to tune thresholds.
- Include clean recordings to measure false alarms.
- Include every flaw class and several severity levels.
- Compare controlled edits vs natural performances where feasible.
- Report failures. Do not cherry-pick.

## Metrics (define computation precisely in code)
- **Event-level P / R / F1** per flaw class. An event matches ground truth if predicted and
  true intervals overlap above an IoU threshold (state the threshold).
- **Temporal localization error**: absolute start error, absolute end error, and interval IoU.
- **False-positive event rate** on clean speech (events per minute of clean audio).
- **Detection rate by class and severity** — plot detection vs severity (the "sweep curve").
- **Calibration** if a defensible confidence is produced.
- **Reproducibility**: same seed/config → same metrics across repeated runs.

## Baselines to beat / compare
1. Rule-based (pause duration + local speech rate) — the honest floor.
2. Feature-distance threshold after speaker normalization.
3. Learned classifier ONLY if enough leak-free data and time exist.

## Honesty rule
A feature changing inside the affected interval is correlation, not proof of cause. Describe the
implemented evidence chain; do not claim "causal detection" beyond what the measurement supports.

## Results

Held-out test split (speaker- AND transcript-disjoint from train): speaker `zira` on
transcripts `tx003`/`tx004`, 26 clips (8 per flaw across severities 0.25/0.5/0.75/1.0 +
2 clean baselines). Thresholds were tuned on the TRAIN split only (`david`,`hazel` ×
`tx001`,`tx002`) with `scripts/tune_thresholds.py`, never on this test set. IoU match
threshold 0.3. Reproduce: `python scripts/run_eval.py` (cached alignments in
`data/align_cache/` make it deterministic).

### Event-level precision / recall / F1

| flaw | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| long_pause | 4 | 0 | 4 | 1.00 | 0.50 | 0.67 |
| rate_deviation | 6 | 0 | 2 | 1.00 | 0.75 | 0.86 |
| monotone | 6 | 0 | 2 | 1.00 | 0.75 | 0.86 |
| **macro-F1** | | | | | | **0.79** |

Every detector has **perfect precision (zero false positives)** on the test set, and
**0.00 false events / minute** on clean speech (0 events over 38.2 s of clean audio).

### Temporal localization error (on matched true positives)

| flaw | n | mean IoU | mean start err (s) | mean end err (s) |
|---|---|---|---|---|
| long_pause | 4 | 0.985 | 0.015 | 0.011 |
| rate_deviation | 6 | 0.770 | 0.202 | 0.139 |
| monotone | 6 | 0.808 | 0.123 | 0.302 |

### Detection rate by severity (right class AND overlaps GT)

| flaw | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|
| long_pause | 0/2 | 0/2 | 2/2 | 2/2 |
| rate_deviation | 0/2 | 2/2 | 2/2 | 2/2 |
| monotone | 0/2 | 2/2 | 2/2 | 2/2 |

No-fabrication guard: **0** evidence objects failed the schema/number check (required 0).

### Tuned thresholds (train-chosen, config.yaml)
- `long_pause_excess_s = 0.2` — train F1 0.86 was flat across 0.1–0.3; chose the
  conservative middle of the tie.
- `rate_speedup_ratio = 0.25`, `rate_min_words = 3` — train F1 0.86 (tie across 0.2–0.25);
  flags a word-run spoken ≥20% faster than the same words in the baseline.
- `monotone_f0_std_ratio = 0.7`, `monotone_min_words = 3` — train F1 0.84 (best; vs 0.80 at
  0.5–0.65); flags a word-run keeping ≤70% of the baseline's pitch variation.

### Failure cases (not cherry-picked)
1. **Subtlest severity (0.25) is missed by all three detectors** (0/2 each). At severity
   0.25 the injected flaw is near the floor of natural variation — a ~0.5 s extra pause, a
   ~12% speed-up, pitch keeping ~75% of its range — and the thresholds that give zero false
   alarms on train cannot separate it from natural speech. This is the precision/recall
   trade-off, chosen honestly on train toward precision.
2. **long_pause recall stalls at 0.50** (misses severities 0.25 and 0.5). `zira`'s baseline
   already contains ~1 s natural pauses, so a severity-0.5 insert (~1 s) does not clear
   `longest natural pause + 0.2 s`. Lowering the margin risks real false positives on
   natural inter-sentence pauses; train could not justify a lower value.
3. **rate localization is looser than long_pause** (IoU 0.77, start err 0.20 s). The
   same-material rule flags a *run of words*; the summed-duration window widens the flagged
   span by up to `min_words-1` words at each edge, so boundaries are word-quantized.
4. **monotone end-boundary error 0.30 s.** Word-final consonants are voiceless and carry no
   pitch, so the last voiced frame of a flagged run lands before the word's true end.

### Interpretation (honesty rule)
These are transparent threshold rules over measured features (pause duration; same-word
forced-alignment durations; same-word pooled F0 std), each emitting a structured
`EvidenceItem` with the baseline value, candidate value, and the rule applied. A feature
moving inside the flagged interval is correlation with the known manipulation, not proof of
perceived-flaw causation; we claim temporal grounding of a measured deviation, not
"causal flaw detection" beyond what the measurement supports.

