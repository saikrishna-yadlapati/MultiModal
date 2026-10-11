# DECISION LOG (ADR-style)

Append-only. Each decision: date, what, why, and whether it's reversible.

## 2026-10-11 — Track C confirmed as working track
What: Build the Track C "Contrastive Speech Analytics & Temporal Flaw Grounding" system.
Why: 75% of rubric is dataset + grounding + features — all controllable and measurable.
Reversible: yes, but switching now is costly. Reconfirm against live Devpost rules before submitting.

## 2026-10-11 — Rate-change uses WSOLA (numpy), not Praat duration resynthesis
What: `rate_deviation` variants are time-stretched with a small in-house WSOLA
(Waveform Similarity Overlap-Add) in `src/speechlab/dataset/variants.py::_wsola`.
Pitch-flatten (`monotone`) still uses Praat PitchTier resynthesis.
Why:
- Session 3 requires "same seed => same output". Praat's *duration* resynthesis
  (`Replace duration tier` + overlap-add) is **not** bit-reproducible — two identical
  runs differed by up to 0.06 in amplitude (verified). Praat's *pitch* resynthesis and
  bare resynthesis ARE bit-exact, so `monotone` keeps using Praat.
- A plain resample would change rate AND pitch (chipmunk effect), conflating two flaw
  classes; WSOLA preserves pitch while changing duration. Verified: a 150 Hz tone sped
  up 2x keeps ~150 Hz, and output is byte-identical across runs.
- librosa (its `time_stretch`) is deferred (numba/py3.13), so WSOLA is ~40 lines of numpy.
Cost: WSOLA is a simpler stretcher than a phase vocoder; mild artifacts on strong
transients are acceptable for a controlled flaw whose ground truth is the timing.
Reversible: yes — swap `_wsola` for librosa later if desired.

## 2026-10-11 — Monotone flattens toward the segment's LOCAL mean F0
What: `flatten_pitch` pulls in-segment pitch toward `Get mean (curve)` over that
segment, not the whole-clip point mean.
Why: forcing pitch to a value outside the segment's natural range makes PSOLA glitch
(octave errors — measured F0 std rose from 8 to 31 Hz). Flattening toward the pitch the
speaker already used stays clean (std 8.08 -> 0.34 at severity 1.0) and is the honest
meaning of "monotone": same phrase, pitch movement removed, not pitch shifted.
Reversible: yes — one line in `flatten_pitch`.

## 2026-10-11 — Dashboard stack = Streamlit (not React+Vite+FastAPI)
What: Build the dashboard in Streamlit (single Python codebase) instead of the brief's
*proposed* (explicitly NOT locked) React+Vite+FastAPI split.
Why:
- Dashboard is only 15% of the score; the win is in the 75% science (data/grounding/features).
- Streamlit = one language, one `pip install`, one `streamlit run` → maximizes reproducibility (10%).
- No second codebase / HTTP layer / build step to maintain under time pressure.
- Has built-in audio, charts, file upload; Plotly works inside it.
Trade-off: less visual polish and control than React. Acceptable for a 15% slice.
Reversible: yes — logic lives in `src/speechlab`, UI is a thin layer; could swap to React+FastAPI later.
Owner can override: say "use React" and we switch the plan.

## 2026-10-11 — Forced aligner = torchaudio MMS_FA (Wav2Vec2 CTC)
What: Use `torchaudio.pipelines.MMS_FA` for word-level forced alignment
(`src/speechlab/alignment/aligner.py`). Chosen + TESTED on the real sample in Session 2.
Why:
- Pip-installable (CPU index), has Python 3.13 wheels, least fragile on Windows.
- aeneas needs eSpeak + ffmpeg + a C-extension build (Windows/py3.13 risk); WhisperX and
  MFA are heavier (whisper/pyannote; conda/kaldi) for no gain over CTC word alignment here.
Cost: pulls torch (~124 MB wheel) + downloads one acoustic model on first use (internet once,
then cached in the torch hub cache, outside the repo).
Evidence: aligned the 41-word baseline sample, all timestamps monotonic and within the
17.43 s duration, mean confidence high (most words 0.9–1.0). See SESSION_LOG Session 2.
Fallback (documented, NOT implemented — aligner works): if the model can't load (offline, no
cache), `align_words` raises RuntimeError; a caller could spread transcript words across the
already-detected voiced regions as a coarse approximation. Not built because YAGNI.
Reversible: yes — alignment is one swappable module behind `align_words()`.

## 2026-10-11 — Manifest storage = stdlib CSV + JSON (no pandas)
What: Manifest is written/read with stdlib `csv` + `json` and a `@dataclass` row
(`src/speechlab/dataset/manifest.py`); list/dict fields are JSON-encoded in CSV cells.
Why: Two rows today, a few hundred at most later. pandas would be a heavy dependency for
I/O that csv does in a few lines. Validator (leakage checks) is plain Python.
Reversible: yes — I/O is isolated in `manifest.py`; swap to pandas if scale ever demands it.

## 2026-10-11 — Product name: NOT chosen
What: No brand yet. Package placeholder = `speechlab`.
Why: "SonicTrace" conflicts with an existing public tool (context §5). Pick + check later.
Reversible: yes (rename is cheap while internal).

## 2026-10-11 — Evidence contract + mechanical no-fabrication guard (Session 4)
What: Detectors never return a bare interval. Each returns an `EvidenceItem`
(`src/speechlab/detection/evidence.py`): flaw_type, interval_s, feature, reference_value,
candidate_value, rule, severity, uncertainty, word_range. Any human-readable sentence is
produced by `phrase_evidence` (deterministic template — no LLM) and must pass
`check_no_fabricated_numbers`, which rejects any number in the prose not present in the
object (or in the code-authored `rule` string, where thresholds live).
Why: This is the project's scientific claim — a number may only come from measurement code,
never from prose or an LLM. The guard makes that enforceable, not a promise: an LLM phraser
can later replace the template behind the same guard and still cannot smuggle in an invented
timestamp, score, or percentage. VERIFIED: 0 fabricated numbers across 12 variants + baseline.
Reversible: yes — phraser is one function behind the guard; the contract is the stable part.

## 2026-10-11 — Detectors are speaker-normalized; thresholds NOT tuned yet (Session 4)
What: All three detectors compare the candidate to the SAME speaker's baseline profile
(`build_reference`: longest natural pause, fastest local rate, typical local F0 std), so they
flag deviation from that speaker, not absolute values. Thresholds in `config.yaml` stay as
documented placeholders — deliberately not tuned.
Why: There is one speaker and all rows are `train`; tuning thresholds now would be tuning on
the only data (leakage against our own honesty rule). Tuning + held-out verification waits for
Session 6, when alignment-based scoring and a real split exist.
Honest findings (Session 4, not cherry-picked): long_pause 4/4 with 0 baseline false positives;
monotone detects the flaw but false-positives on natural low-variability windows (needs
word-aligned same-material comparison); rate_deviation 0/4 (energy-peak syllable proxy fails on
real speech, needs alignment-based rate). The monotone/rate fixes depend on the Session 2
aligner and are scheduled for Session 6.
Reversible: yes — thresholds are config; detectors are swappable modules.
