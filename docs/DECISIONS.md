# DECISION LOG (ADR-style)

Append-only. Each decision: date, what, why, and whether it's reversible.

## 2026-10-11 — Track C confirmed as working track
What: Build the Track C "Contrastive Speech Analytics & Temporal Flaw Grounding" system.
Why: 75% of rubric is dataset + grounding + features — all controllable and measurable.
Reversible: yes, but switching now is costly. Reconfirm against live Devpost rules before submitting.

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

## 2026-10-11 — Forced aligner: NOT chosen yet
What: Defer aligner choice to Session 2; test one on a single sample before committing.
Why: Aligners are heavy; brief says prove core before installing a big stack.
Candidates: torchaudio CTC, WhisperX, aeneas, MFA (see `ARCHITECTURE.md`).
Reversible: yes.

## 2026-10-11 — Product name: NOT chosen
What: No brand yet. Package placeholder = `speechlab`.
Why: "SonicTrace" conflicts with an existing public tool (context §5). Pick + check later.
Reversible: yes (rename is cheap while internal).
