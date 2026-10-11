# SESSION LOG

One entry per working session: what changed, why, how it was tested, the actual result,
limitations, next step, and the commit. No fabricated results.

---

## Session 0 — 2026-10-11 — Scaffold & planning
**Goal:** Turn the context-only folder into an organized full-scale project skeleton with docs,
memory, dependency plan, and a locked session roadmap.

**Files changed/created:**
- `docs/PLAN.md`, `docs/ARCHITECTURE.md`, `docs/DATASET_CARD.md`, `docs/EVALUATION.md`,
  `docs/DECISIONS.md`, `docs/SESSION_LOG.md`
- `README.md`, `requirements.txt`, `.gitignore`, `config.yaml`
- `src/speechlab/` package skeleton (empty modules with docstrings)
- `data/`, `tests/`, `scripts/` directories with `.gitkeep`
- Memory files under the Claude memory directory (user, project, stack, workflow)

**Why:** Owner wants a full end-to-end project built across multiple sessions with a meaningful
commit per session, and wants to understand the system while it's built.

**Decisions made:** Streamlit for dashboard; aligner + product name deferred (see `DECISIONS.md`).

**How to test (Session 0 acceptance):**
- Review the directory tree and docs for clarity.
- `python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt`
  (expected: installs; audio libs are the risk — report any failure, we fix in Session 1).

**Actual result:** scaffolding created. Dependency install NOT yet run/verified in this session — marked untested.

**Limitations:** No code logic yet, no data, no aligner, no metrics. All real work starts Session 1.

**Next step:** Session 1 — one-sample vertical slice (features + blind flaw detection).

**Suggested commit:** `chore: scaffold repo structure, planning docs, and session roadmap`

---

## Session 1 — 2026-10-11 — One-sample vertical slice (features + blind flaw detection)
**Goal:** Prove one audio sample end to end: synthesize a baseline with known transcript,
extract pause/speech-rate/F0 features, inject a known pause into a copy, detect it *blind*,
and score the predicted interval against ground truth.

**Files created:**
- `scripts/make_sample.py` — native Windows TTS (SAPI via PowerShell) → `data/raw/baseline_001.{wav,txt}`.
- `scripts/run_slice.py` — the end-to-end slice (load → features → inject → blind detect → score).
- `src/speechlab/features/pauses.py` — RMS energy + silence/pause detection (pure numpy).
- `src/speechlab/features/f0.py` — F0 contour + stats via Praat/parselmouth.
- `src/speechlab/features/rate.py` — global speech rate (words per minute of voiced time).
- `src/speechlab/dataset/inject.py` — `inject_pause` (returns exact ground-truth interval).
- `src/speechlab/detection/pause_flaw.py` — blind long-pause detector (absolute threshold).
- `src/speechlab/evaluation/intervals.py` — `interval_iou`, `start_end_error` (reused in Session 6).
- `tests/test_slice.py` — 3 self-checks (pure numpy, no SAPI).
- `requirements-lock.txt` — `pip freeze` of the verified environment.

**Files changed:**
- `config.yaml` — added `features.silence_rel_db` (-35 dB) and `features.min_pause_s` (0.15 s).
- `requirements.txt` — corrected the broken pins (see "Decisions/findings" below).

**Why:** Milestone 1 of the plan. Prove the core measurement+localization loop on one
sample before adding alignment, a dataset, or a dashboard.

**Decisions / findings (environment):**
- The old `requirements.txt` pinned `numpy<2.0` + `librosa==0.10.2`. **This is unbuildable
  on Python 3.13** (no numpy 1.x wheel for 3.13; librosa's numba conflicts with numpy 2).
  Resolution: Session 1 uses **numpy 2.5.3 + scipy + soundfile + parselmouth 0.4.7**, which
  install cleanly on Windows/py3.13/CPU. **librosa is deferred** to Session 4 (MFCC) and will
  need `numba>=0.61`. F0 uses Praat (parselmouth), which is self-contained and avoids the fight.

**How to test (Session 1 acceptance):**
```
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt
python -m pytest tests/test_slice.py -q
python scripts/make_sample.py
python scripts/run_slice.py data/raw/baseline_001.wav data/raw/baseline_001.txt
```

**Actual result (VERIFIED this session):**
- `pytest tests/test_slice.py` → **3 passed** in 0.15s.
- `make_sample.py` → produced a 17.4 s, 22050 Hz WAV (41-word transcript).
- `run_slice.py` → baseline F0 mean/std ≈ 93.0 / 23.4 Hz, global rate ≈ 203 wpm.
  Injected a 1.5 s pause at 4.5 s (ground truth [4.500 → 6.000]). Blind detector predicted
  **[4.507 → 5.984]**, **IoU 0.984**, start/end error **7 ms / 16 ms** → **PASS** (threshold 0.3).

**Limitations (honest):**
- The absolute-threshold detector also flagged **4 natural TTS inter-sentence pauses** (≈0.9 s)
  as "long pauses" — i.e. false positives. The fix is baseline-relative comparison after
  forced alignment (Sessions 2/4), not a higher threshold.
- Speech rate is **global only**; local/per-word rate needs alignment (Session 2).
- Injected silence = pure zeros; a computer-injected pause is clean ground truth but is **not**
  proof the detector handles naturally-occurring human pauses. Natural examples come later.
- One speaker (SAPI default voice), one sample. No speaker normalization yet.

**Next step:** Session 2 — dataset manifest schema + pick/test ONE forced aligner on this sample
(or document a fallback), enabling baseline-relative detection and local rate.

**Suggested commit:** `feat: end-to-end single-sample feature extraction + blind pause detection`

---

## Session 2 — 2026-10-11 — Dataset manifest + forced-alignment decision
**Goal:** (A) Define a dataset manifest schema with stable IDs, CSV I/O, and a validator that
blocks train/test leakage; seed it from the real Session 1 sample. (B) Pick and TEST one forced
aligner on a single sample (or document a fallback).

**Files created:**
- `src/speechlab/dataset/manifest.py` — `ManifestRow` dataclass (schema == `DATASET_CARD.md`),
  `make_pair_id`/`make_sample_id`, `write_manifest`/`read_manifest` (stdlib csv+json),
  and `validate_manifest` (leakage + ground-truth + enum checks). Has a `__main__` self-check.
- `src/speechlab/alignment/aligner.py` — `align_words()` via torchaudio **MMS_FA** (Wav2Vec2 CTC):
  normalize text to the model vocab, resample to 16 kHz, return per-word `[start,end]` + score.
- `scripts/build_manifest.py` — seeds `data/manifest/manifest.csv` with 2 truthful rows
  (baseline + injected-pause variant); sample rate and GT interval derived from the real files.
- `scripts/run_align.py` — aligns one sample, prints word times, sanity-checks them.
- `tests/test_manifest.py` — 4 self-checks (valid round-trip, missing GT, speaker leakage, pair split).

**Files changed:**
- `requirements.txt` — manifest uses stdlib (no pandas); torch/torchaudio added with CPU-index note.
- `requirements-lock.txt` — re-frozen (adds torch 2.14.1+cpu, torchaudio 2.11.0+cpu + deps).
- `docs/DECISIONS.md` — aligner = torchaudio MMS_FA; manifest = stdlib CSV/JSON (both reversible).

**Decisions:** Forced aligner = **torchaudio MMS_FA** (pip-only, py3.13 wheels, least fragile on
Windows vs aeneas/WhisperX/MFA). Manifest = stdlib CSV/JSON (no pandas) — see `DECISIONS.md`.

**How to test (Session 2 acceptance):**
```
.venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
.venv\Scripts\python -m pytest tests/test_manifest.py -q
.venv\Scripts\python scripts/build_manifest.py
.venv\Scripts\python scripts/run_align.py data/raw/baseline_001.wav data/raw/baseline_001.txt
```

**Actual result (VERIFIED this session):**
- `pytest tests/test_manifest.py` → **4 passed**; full suite **7 passed**.
- `build_manifest.py` → wrote **2 rows** to `data/manifest/manifest.csv`; validator clean;
  sr=22050, GT pause interval [4.500 → 6.000].
- `run_align.py` → downloaded the MMS_FA model (first run), aligned all **41 words**,
  timestamps **monotonic and within the 17.43 s duration**, mean confidence high
  (most words 0.90–1.00; lowest 0.16 on "everyone"). **RESULT: PASS**. Word gaps fall at
  the real inter-sentence pauses (e.g. "communication" ends 6.10 s, next word 7.08 s).

**Limitations (honest):**
- Alignment is validated only for *sane* timestamps (monotonic, in-range), not against
  hand-labeled truth — no gold word boundaries exist yet, so word-level alignment error is
  unmeasured. Confidence scores come from the model, not an independent check.
- First alignment run needs internet to fetch the model; offline runs fail (fallback documented
  in `DECISIONS.md`, not implemented since the aligner works).
- Manifest holds the single Session 1 synthetic pair; still one speaker, split all `train`.
- torch adds ~124 MB; this is the project's first heavy dependency (owner-approved).

**Next step:** Session 3 — seeded controlled variant generators (pause / local-rate / pitch-flatten)
with a severity sweep, each writing GT intervals + params into the manifest.

**Suggested commit:** `feat: dataset manifest schema and torchaudio forced-alignment integration`

---

## Session 3 — 2026-10-11 — Controlled contrastive variant generators + severity sweep
**Goal:** Build deterministic (seeded) generators for the three MVP flaw classes
(long_pause, rate_deviation, monotone), each producing a severity gradient, writing its
exact ground-truth interval + transform params to the manifest, and reproducible
byte-for-byte on re-run.

**Files created:**
- `src/speechlab/dataset/variants.py` — `change_rate` (WSOLA time-stretch of one segment,
  pitch preserved, pure numpy), `flatten_pitch` (Praat PitchTier flatten toward the
  segment's local mean), `_wsola` helper; re-exports `inject_pause`. Has a `__main__` self-check.
- `scripts/make_variants.py` — seeded generator: for the one baseline, sweeps 4 severities
  × 3 flaw types = 12 variants, writes WAVs + rebuilds the manifest (baseline + variants),
  validates it. Supersedes `scripts/build_manifest.py`'s 2-row seed.
- `tests/test_variants.py` — 5 self-checks (pure/local, no SAPI, no model download).

**Files changed:**
- `config.yaml` — added a `variants:` block (severity grids + the knobs mapping severity →
  pause length / rate factor / flatten amount, plus segment length & safe-region bounds).

**Why:** Milestone 3 (data engineering & stress testing, 30% of rubric). A controlled
severity spectrum with known ground truth is what lets Session 6 measure detection-vs-severity.

**Decisions / findings:**
- Praat's **duration** resynthesis is NOT bit-reproducible (two runs differed up to 0.06
  amplitude) — so rate uses an in-house **WSOLA** (numpy, bit-exact, pitch-preserving).
  Praat's **pitch** resynthesis IS bit-exact, so monotone keeps using Praat. (`DECISIONS.md`)
- Monotone must flatten toward the segment's **local** mean F0; forcing the global mean
  pushed pitch out of range and made PSOLA glitch (measured F0 std 8 → 31 Hz). (`DECISIONS.md`)

**How to test (Session 3 acceptance):**
```
.venv\Scripts\python -m pytest tests/test_variants.py -q
.venv\Scripts\python scripts/make_variants.py
```

**Actual result (VERIFIED this session):**
- `pytest` → **12 passed** (full suite; 5 new).
- `make_variants.py` → wrote **13 rows** (1 baseline + 12 variants) to
  `data/manifest/manifest.csv`; validator clean (no leakage, every variant has GT + params).
  Seeded locations (seed 42): pause@9.949s, rate@2.261s, monotone@4.869s, 2.0 s segments.
- **Reproducible:** two consecutive `make_variants.py` runs produced **byte-identical** WAVs
  (sha256 match across all 12 variants).
- **Flaws are real and scale with severity (measured on the generated files):**
  - long_pause: clip duration 17.432 s → 17.932 / 18.432 / 18.932 / 19.432 s (sev 0.25→1.0;
    +0.5 → +2.0 s silence inserted).
  - rate_deviation (sev 1.0): the 2.0 s segment → 1.0 s (factor 0.5, clip 17.432 → 16.432 s);
    pitch preserved (test: sped-up tone keeps ~150 Hz).
  - monotone: in-segment F0 std 8.08 Hz (baseline) → **6.08 / 4.02 / 2.02 / 0.34 Hz**
    (sev 0.25→1.0) — a clean monotonic gradient; mean pitch unchanged.

**Limitations (honest):**
- Still ONE speaker, ONE transcript (single synthetic baseline). The variant set is a
  severity spectrum over that one pair; splits are all `train`. Real external-validity needs
  more speakers / natural recordings (later).
- WSOLA is a basic stretcher; strong transients can get mild artifacts (acceptable — GT is
  the timing). `flatten_pitch` raises if a chosen segment is mostly unvoiced (fail-loud, not
  silently mislabeled).
- These are computer-injected flaws = ground truth for *that manipulation only*, not proof
  the detector handles naturally-occurring human delivery flaws.
- Variant locations are seeded-random within a safe region; they are not yet guaranteed to
  avoid natural pauses (monotone@4.869s spans a sentence boundary and still flattens cleanly).

**Next step:** Session 4 — transparent baseline detectors (pause / rate / monotone) with
speaker-aware normalization + strict evidence objects; run them blind on this variant set.

**Suggested commit:** `feat: controlled contrastive variant generators with severity sweep`

---

## Session 4 — 2026-10-11 — Baseline detectors + structured evidence objects
**Goal:** Build the three transparent, speaker-normalized blind detectors (long_pause,
rate_deviation, monotone) that emit a STRICT evidence object instead of a bare interval,
plus a mechanical guard proving generated prose contains no number the object does not hold.

**Files created:**
- `src/speechlab/detection/evidence.py` — `EvidenceItem` dataclass (== `ARCHITECTURE.md`
  contract), `validate_evidence` (schema), `check_no_fabricated_numbers` (the guard:
  returns any number in a sentence not backed by the object), `phrase_evidence`
  (deterministic, number-safe template phraser — no LLM). Has a `__main__` self-check.
- `src/speechlab/features/local.py` — windowed features: `local_f0_std` (pitch
  variability per sliding window) and `local_syllable_rate` (energy-peak syllable proxy,
  no alignment needed). `__main__` self-check.
- `src/speechlab/detection/detectors.py` — `build_reference` (measures the baseline once:
  longest natural pause, fastest local rate, typical local F0 std — all speaker-specific)
  + three blind detectors + `detect_all`. Each returns `EvidenceItem`s.
- `scripts/run_detect.py` — builds the reference from the baseline, runs all detectors
  BLIND over every manifest variant, prints each evidence object + its phrasing + the
  guard result, then reveals ground truth for an honest detection-vs-severity table.
- `tests/test_evidence.py` — 7 self-checks (evidence validates; guard catches an invented
  number; schema rejects bad intervals/empty uncertainty; each detector fires on its
  synthetic flaw; `detect_all` emits only valid, number-safe evidence).

**Files changed:**
- `config.yaml` — `detection` block documented as speaker-normalized deviation knobs;
  added `long_pause_excess_s` (0.3), `rate_window_s`/`rate_hop_s`, `monotone_window_s`/
  `monotone_hop_s`. Left `long_pause_s` (Session 1 slice still uses it).
- `docs/DECISIONS.md` — evidence contract + no-fabrication guard decision.

**Why:** Milestone 4 (25% temporal grounding + 20% feature extraction). This is where the
project's scientific claim becomes real: numbers come from code, the LLM may only phrase a
verified object, and a check enforces it.

**How to test (Session 4 acceptance):**
```
.venv\Scripts\python -m pytest tests/test_evidence.py -q
.venv\Scripts\python scripts/run_detect.py
```

**Actual result (VERIFIED this session):**
- `pytest` → **19 passed** (full suite; 7 new).
- **Evidence guard: 0 fabricated numbers** across all 12 variants + baseline; every emitted
  evidence object passed the schema. This is the core acceptance — PASS.
- Reference profile (measured on the real baseline): longest natural pause 0.92 s,
  fastest local rate 6.00 syll/s, typical local F0 std 9.93 Hz.
- Blind detection-vs-severity (right class AND overlaps ground truth):
  - **long_pause 4/4** — IoU rises with severity 0.36 → 0.53 → 0.63 → 0.69. **0 false
    positives on the clean baseline.**
  - **monotone 4/4** detected the flattened segment, BUT the detector also fires **4 false
    positives on the clean baseline** (natural phrase-end pitch dips, local F0 std 0.39–2.34
    Hz, below 0.5× the 9.93 Hz median).
  - **rate_deviation 0/4** — the energy-peak syllable proxy did NOT fire on any rushed
    segment of real SAPI speech (did not clear the baseline's fastest natural window +35%).
  - Overall: 8/12 right-class-and-overlapping.

**Limitations (honest — not cherry-picked):**
- **Monotone over-fires**: comparing each window to the clip-wide median flags natural
  low-variability moments as "monotone". A real fix needs word-aligned baseline-vs-candidate
  comparison (same material), i.e. the aligner from Session 2 — deferred to Session 6.
- **Rate detector failed on real speech**: the energy-peak proxy is too noisy / the 2 s
  segment is diluted by the 1 s window. Needs alignment-based local rate (words ÷ aligned
  duration) — Session 6. Marked `ponytail:` in `features/local.py`.
- Thresholds are still placeholders. They are NOT tuned here on purpose — there is one
  speaker, all `train`; tuning now would be tuning on the only data. Tuning + held-out
  verification is Session 6.
- Absolute-time IoU understates localization because pause/rate transforms shift later
  timestamps; alignment-based scoring comes in Session 6.

**Next step:** Session 5 — Streamlit dashboard (upload/select, playback, synced timelines,
highlighted intervals from these evidence objects, limitations notice). The detector fixes
(monotone normalization, rate rebuild, threshold tuning) land in Session 6 with the aligner
and the held-out split.

**Suggested commit:** `feat: baseline detectors and structured evidence objects`

---

## Session 5 — Interactive Streamlit dashboard (2026-10-11) — DONE / VERIFIED

**Goal:** A dashboard that loads a speech clip, analyzes it blind against this speaker's
baseline, and *shows* the detected flaw intervals on a timeline — with the honesty guard
carried all the way to the screen.

**Acceptance test:** `streamlit run src/speechlab/app/main.py` loads, analyzes a sample, and
shows the detected intervals; headless `AppTest` boots the script with no exception and renders
the metrics + evidence. **PASS.**

**What was built:**
- `src/speechlab/app/analysis.py` — pure analysis layer (NO `st.*`): `load_audio` (path/bytes/
  file-like), `rms_series`/`f0_series` (plot DataFrames reusing the SAME feature code the
  detectors see), `evidence_to_frame`, `heuristic_score`, and `analyze(cfg, base_y, bsr, cand_y,
  csr) -> dict`. `analyze` re-runs the Session 4 guards (`validate_evidence`,
  `check_no_fabricated_numbers`) so the UI can display a trust badge.
- `src/speechlab/app/main.py` — thin Streamlit script: sidebar source picker (dataset variant or
  WAV upload), baseline + candidate players, loudness timeline with shaded detected intervals
  (Altair layered `mark_rect`), voiced-pitch contour, per-flaw evidence cards, "Reveal ground
  truth" dashed overlay (blind detector never saw it), limitations expander, guard trust badge.
- `tests/test_dashboard.py` — 4 tests (3 pure-logic pytest + 1 headless `AppTest`).

**Verified results:**
- Full suite **23 passed** (19 prior + 4 new) in ~3 s.
- Pure-logic: score formula is exactly `max(0, 100 - 8*n)`; on a synthetic baseline + injected
  2 s pause, `analyze` catches a `long_pause`, score drops, and **both guards clean**
  (0 fabricated numbers, 0 schema errors).
- Headless UI: `AppTest.from_file(main.py).run()` → no exception; 4 KPI metrics render;
  guard-success badge shows (not the error badge); "Evidence" section present.

**Decisions recorded in docs/DECISIONS.md:** charts use Altair (bundled, no new dep) instead of
Plotly; dashboard split into pure `analysis.py` + thin `main.py` so the no-fabrication contract
and unit-testability both hold.

**Limitations (honest):**
- Everything the detectors show inherits Session 4's limits — one speaker, computer-injected
  flaws, untuned placeholder thresholds, monotone over-fires on natural low-variability windows,
  rate_deviation 0/4 on real speech. The dashboard displays these faithfully; it does not fix them.
- The delivery-concern score is an illustrative transparent formula, NOT a validated measure of
  speaking quality (shown framed as such in the UI).
- `AppTest` cannot simulate on-chart selection / real browser rendering; the headless test covers
  "script runs and shows the right elements", not pixel layout.

**Next step:** Session 6 — alignment-based fixes (monotone normalization + rate rebuild on the
Session 2 aligner), threshold tuning, and a real held-out split with reported metrics.

**Suggested commit:** `feat: interactive Streamlit dashboard`

---

## Session 6a — 2026-10-11 — Multi-speaker/transcript dataset + held-out split
**Goal:** Sessions 1–5 ran on ONE speaker / ONE transcript (all `train`), so a genuine
held-out split did not exist and Session 6's metrics could not be honest (tuning + testing
on the only data = leakage). Expand the dataset to multiple speakers and transcripts and
assign a speaker- AND transcript-disjoint train/test split, so Session 6b can tune on train
and report on an unseen speaker + unseen words. (Owner chose this over a single-speaker
"dev-set" eval.)

**Files created:**
- `src/speechlab/dataset/plan.py` — PURE dataset definition (no audio/SAPI/model): the 3
  SAPI voices, 4 transcripts, and the disjoint split policy, plus `planned_pairs()`. One
  importable source of truth shared by the generator and the split test.
- `scripts/make_dataset.py` — synthesizes a baseline per (speaker, transcript) with the
  explicit SAPI voice, runs the Session 3 severity sweep for each, writes one manifest with
  the split, validates it, and prints sha256s. Supersedes `scripts/make_variants.py`.
- `tests/test_dataset_split.py` — 5 pure tests: plan disjoint on both axes; both splits
  present with known voices/transcripts; every pair's split agrees with both sets; a
  plan-shaped manifest passes the leakage validator; forcing a test speaker into train is
  caught as LEAKAGE.

**Files changed:**
- `src/speechlab/app/main.py` — **bug fix exposed by multi-speaker data**: the dashboard now
  picks the candidate's OWN pair baseline (same speaker+transcript) as the reference instead
  of `baselines[0]`. With one speaker that was always correct; with many it was comparing a
  Zira clip against David's baseline (cross-speaker) and defeating speaker normalization.
  Updated the stale "run make_variants.py" hint to make_dataset.py.

**Why:** Milestone 6 prerequisite + more 30%-rubric data engineering. A disjoint split is the
only honest basis for the Session 6b metrics.

**How to test (Session 6a acceptance):**
```
.venv\Scripts\python -m pytest tests/test_dataset_split.py -q
.venv\Scripts\python scripts/make_dataset.py
```

**Actual result (VERIFIED this session):**
- `pytest tests/test_dataset_split.py` → **5 passed**; full suite **28 passed** (23 prior + 5).
- `make_dataset.py` → wrote **78 rows** (6 baselines + 72 variants) to
  `data/manifest/manifest.csv`; validator **clean**; split = **52 train / 26 test**,
  speaker- and transcript-disjoint (train = David/Hazel × tx001/tx002;
  test = Zira × tx003/tx004).
- **Reproducible:** two consecutive runs produced **byte-identical** WAVs (sha256 match on
  all 6 baselines + 72 variants) — SAPI synth is deterministic on this machine.
- Dashboard fix verified by the existing headless `AppTest` still booting clean.

**Limitations (honest):**
- Three MACHINE voices are still not natural human speakers; a disjoint split proves
  generalization to an unseen SYNTHETIC speaker, not to real human delivery. Real recordings
  remain future work and are stated as a limitation.
- Flaws are still computer-injected (ground truth for the manipulation only).
- Reproducibility is scoped to "same machine, same SAPI voices"; a different OS/voice-engine
  version could change the synthesized bytes (and so the seeded flaw locations).
- No metrics yet — detectors are unchanged (monotone over-fires, rate_deviation 0/4 from
  Session 4); fixes + held-out numbers are Session 6b.

**Next step:** Session 6b — fix monotone (word-aligned comparison) and rate_deviation
(alignment-based local rate) on the Session 2 aligner, tune thresholds on TRAIN only, and
report real held-out metrics on the TEST set.

**Suggested commit:** `feat: multi-speaker/transcript dataset with held-out split`

---

## Session 6b — honest detector fixes + tuned held-out metrics (VERIFIED)

**Files changed:**
- `src/speechlab/features/local.py` — added `same_material_rate_regions()` (summed-window
  run finder: candidate vs baseline word durations for the SAME words) and
  `same_material_monotone_regions()` + `_pooled_f0_std()` (candidate vs baseline pooled F0
  std over the SAME words). The Session 4 energy-proxy / typical-median helpers remain for
  the no-transcript fallback.
- `src/speechlab/detection/detectors.py` — `detect_rate_deviation` and `detect_monotone`
  now use same-material comparison when baseline+candidate forced-alignment spans are
  present (fall back to the old rules otherwise). `build_reference` stores the baseline
  word spans and baseline F0 contour. `detect_all` passes spans to both detectors.
- `config.yaml` — new honest-rule knobs; thresholds TUNED ON TRAIN (see below).
- `scripts/tune_thresholds.py` — NEW. Sweeps each flaw's knobs on the TRAIN split only,
  scores F1 (negatives = clean + other-flaw variants), prints the winner. Never sees test.
- `scripts/align_one.py` — NEW. Aligns one clip per process (the CPU torch aligner
  segfaults when many alignments share a process; one-per-process via subprocess is stable).
- `tests/test_rate.py`, `tests/test_monotone.py` — pure self-checks for the two new rules.
- `docs/EVALUATION.md` — Results section filled with the real held-out numbers + 4 failures.

**Why:** Session 4 detectors were dishonest/broken (rate_deviation 0/4; monotone 123 FP).
The same-material idea — compare each word to how the SAME speaker said the SAME word in
the baseline — cancels natural speed/prosody and is the honest fix. Milestone 6b requires
real held-out metrics with thresholds tuned only on train.

**How to run/test:**
```
.venv\Scripts\python -m pytest tests/ -q
.venv\Scripts\python scripts/tune_thresholds.py     # train-only threshold search
.venv\Scripts\python scripts/run_eval.py            # held-out metrics on test split
```

**Actual result (VERIFIED this session):**
- Full suite **41 passed** (28 prior + 13 new rate/monotone/metrics checks).
- Train tuning winners: long_pause excess=0.2 (F1 0.86, tie 0.1–0.3); rate speedup=0.25,
  min_words=3 (F1 0.86); monotone ratio=0.7, min_words=3 (F1 0.84).
- Held-out test (thresholds NOT tuned on test): macro-F1 **0.79**.
  - long_pause P1.00 R0.50 F1 0.67 (IoU 0.985)
  - rate_deviation P1.00 R0.75 F1 0.86 (IoU 0.770) — was 0/8 before the fix
  - monotone P1.00 R0.75 F1 0.86 (IoU 0.808) — was 123 FP / P0.06 before the fix
  - **0.00 false events / minute** on clean speech; no-fabrication guard **0** failures.
- All misses are the subtlest severity (0.25); long_pause also misses 0.5. Reported in
  EVALUATION.md, not cherry-picked.

**Limitations (honest):**
- Severity-0.25 flaws are below the zero-false-alarm threshold for every detector; recall is
  bought with precision, chosen on train toward precision.
- Still synthetic speakers + computer-injected flaws (ground truth is the manipulation).
- Rate/monotone run boundaries are word-quantized (looser IoU than long_pause).
- CPU torch aligner segfaults on repeated in-process alignment; worked around with
  one-alignment-per-process. Alignments are cached as small JSON span files in
  `data/align_cache/` — recommend committing them so the held-out numbers reproduce
  without re-running the flaky aligner (owner's call on repo hygiene).

**Next step:** Session 7 — surface the EvidenceItem + localization in the Streamlit
dashboard (the 15% rubric slice), or add a confidence/calibration signal.

**Suggested commit:** `feat: honest same-material rate/monotone detectors + held-out metrics`

---
