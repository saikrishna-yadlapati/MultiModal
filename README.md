# Contrastive Speech Analytics & Temporal Flaw Grounding
_Multimodal AI Hackathon 2026 — Track C. Working name: TBD._

A speech-delivery debugging system. It builds controlled contrastive pairs of
"ideal" vs "flawed" speech, locates each flaw **in time, blind** (the detector never sees the
answer key), explains it from **measured acoustic evidence** (not LLM guesses), and reports
**honest held-out metrics** on an unseen speaker speaking unseen sentences.

> Status: working end-to-end pipeline (Sessions 0–6b complete and verified). Full test
> suite: **41 passed**. Held-out evaluation done. See `docs/SESSION_LOG.md` for every step.

## Why it's not just "ask an LLM"
- **Same transcript**, controlled delivery edits with logged parameters → **known
  ground-truth intervals**. We know exactly where the flaw is and how it was made.
- The detector sees **audio only**, predicts the flawed interval, then we reveal the label
  and measure the error. That is temporal grounding, not vibes.
- Every explanation is built from a structured **evidence object** (measured baseline value,
  measured candidate value, the rule that fired). A mechanical guard rejects any number in the
  prose that measurement code did not produce — so an LLM can phrase findings but **cannot
  invent a timestamp, score, or percentage**.

## Headline results (held-out test set — real, not cherry-picked)
Test = speaker `zira` on transcripts `tx003`/`tx004` (speaker- AND transcript-disjoint from
train). Thresholds tuned on the TRAIN split only. IoU match threshold 0.3. Full table +
failure cases in `docs/EVALUATION.md`.

| flaw | precision | recall | F1 | mean IoU |
|---|---|---|---|---|
| long_pause | 1.00 | 0.50 | 0.67 | 0.985 |
| rate_deviation | 1.00 | 0.75 | 0.86 | 0.770 |
| monotone | 1.00 | 0.75 | 0.86 | 0.808 |
| **macro-F1** | | | **0.79** | |

- **Zero false positives** on every flaw; **0.00 false events / minute** on clean speech.
- No-fabrication guard: **0** evidence objects failed the number check.
- Honest miss: the subtlest severity (0.25) is below the zero-false-alarm threshold for all
  three detectors — reported, not hidden.

## Project layout
```
docs/              planning, architecture, dataset card, evaluation, decisions, session log
src/speechlab/     importable logic:
  features/        pauses, F0 (Praat), speech rate, windowed + same-material features
  alignment/       torchaudio MMS_FA forced aligner + span cache
  dataset/         manifest schema, split plan, controlled variant generators
  detection/       speaker-normalized blind detectors + evidence objects + no-fabrication guard
  evaluation/      interval IoU / errors, held-out metrics
  app/             Streamlit dashboard (pure analysis.py + thin main.py)
scripts/           command-line entry points
data/              raw baselines, variants, manifest.csv, cached alignment spans
tests/             41 self-checks
```

## Setup (Windows / Python 3.13 / CPU — verified)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# torch/torchaudio are CPU wheels from the PyTorch index (NOT PyPI):
pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
```
Exact pinned versions that pass here are in `requirements-lock.txt` (`pip install -r
requirements-lock.txt` to reproduce the verified environment byte-for-byte).

The forced aligner downloads one acoustic model on first use (needs internet once, then
cached). Cached alignment spans are committed in `data/align_cache/`, so `run_eval.py`
reproduces the held-out numbers **without** re-running the aligner.

## Run
```bash
python -m pytest tests/ -q                 # 41 self-checks (~2 s)
python scripts/make_dataset.py             # rebuild dataset: 6 baselines + 72 variants, 78-row manifest
python scripts/tune_thresholds.py          # threshold search on TRAIN split only
python scripts/run_eval.py                 # held-out metrics on the TEST split
streamlit run src/speechlab/app/main.py    # interactive dashboard
python scripts/run_slice.py data/raw/baseline_001.wav data/raw/baseline_001.txt   # single-sample demo
```

## Reproducibility
- One seed (`config.yaml: seed: 42`) drives all randomness; same seed → same flaw locations.
- Two dataset-generation runs produce **byte-identical** WAVs (sha256 match, verified).
- Thresholds live in `config.yaml` (documented, tuned on train only) — the control panel.
- Metrics come only from `scripts/run_eval.py`; no number in the docs is hand-typed.

## Dataset & licensing
Current dataset is **synthetic** (3 Windows SAPI voices × 4 transcripts), so no third-party
audio is redistributed — fully reproducible from `scripts/make_dataset.py`. Policy, schema,
and the honest external-validity limitation are in `docs/DATASET_CARD.md`. No
non-redistributable copyrighted audio is committed.

## Limitations (honest)
- Speakers are machine (SAPI) voices, not natural humans; flaws are computer-injected (ground
  truth for the manipulation only, not proof of naturally-occurring human delivery flaws).
- Severity-0.25 flaws are missed by all detectors (precision/recall trade-off, chosen on train).
- Rate/monotone interval boundaries are word-quantized (looser IoU than long_pause).
- The dashboard's "delivery-concern score" is a documented constant formula, **not** a
  validated quality metric; it is labeled as such in the UI.

## AI-tool disclosure
Built with help from AI coding assistants (primarily Claude Code; review via Codex / Google
Antigravity where used). AI assisted implementation and explanation. **All metrics and results
are produced by the code and verified by the team, not generated by an LLM** — this is enforced
mechanically by the no-fabrication guard in `src/speechlab/detection/evidence.py`.

## Documentation
Start with `HACKATHON_CONTEXT.md` (master brief), then `docs/ARCHITECTURE.md` (data flow +
evidence contract), `docs/EVALUATION.md` (results), and `docs/DECISIONS.md` (why each choice).
