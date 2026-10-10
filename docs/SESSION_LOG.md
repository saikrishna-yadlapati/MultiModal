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
