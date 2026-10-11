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
