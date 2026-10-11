# New-chat kickoff prompt (reusable every session)

Open this file, copy the block below, paste it as the first message in a new Claude chat.
It is session-agnostic: it tells the agent to read `docs/SESSION_LOG.md` and continue from the
last completed session. You do **not** need to edit it each time.

(Optional) If you want to force a specific session or add a note, type it after the block.

---

```
Continuing the Multimodal AI Hackathon Track C project (contrastive speech analytics &
temporal flaw-grounding system). I build it across multiple sessions and commit manually
after each one. Deadline is deprioritized — maximize quality and scientific honesty.

STEP 1 — Read these in full before doing or changing anything:
  - HACKATHON_CONTEXT.md   (master source of truth: event, Track C, rubric, guardrails)
  - CLAUDE.md              (project rules for you)
  - docs/PLAN.md           (session roadmap + each session's acceptance test + commit)
  - docs/ARCHITECTURE.md   (data flow, modules, evidence-object contract)
  - docs/SESSION_LOG.md    (what is DONE and verified — read the LAST entry to find where we are)
  - docs/DECISIONS.md      (locked decisions; don't re-litigate them)
  - docs/EXPLANATION_STYLE.md (HOW to explain to me every session — teach as you build, define every term)

STEP 2 — Inspect the actual repo (files, src/speechlab/, scripts/, tests/, config.yaml,
requirements-lock.txt) and confirm current state against the session log.

STEP 3 — Tell me, in 3-4 lines: the current phase, what is already done and verified, and
the single next session to run (the first session in docs/PLAN.md not yet marked done in
docs/SESSION_LOG.md). Then start that session.

ENVIRONMENT (verified, do not re-litigate): Windows / Python 3.13.2 / CPU. Use the existing
.venv (.venv/Scripts/python). Working deps are frozen in requirements-lock.txt. librosa is
DEFERRED to the MFCC session (unbuildable on py3.13 + numpy 2 until numba>=0.61); F0 uses
praat-parselmouth. Don't install heavy libraries before testing them on one sample.

HOW YOU MUST WORK (from CLAUDE.md):
  - Small, testable steps. State the goal and the acceptance test BEFORE coding.
  - Prove it on one real sample. Never fabricate metrics/results. Mark untested as untested.
  - Never let an LLM invent numbers; explanations come from measured features/evidence objects.
  - Watch for leakage (speaker/transcript), timestamp errors, privacy, audio licensing,
    reproducibility, and Windows/py3.13 runtime compatibility.
  - Don't change track/architecture/scope/deps significantly without explaining and asking.

END OF SESSION — give me:
  1. Files changed and why.
  2. How to run/test it (exact commands).
  3. The ACTUAL test result (real output, pass/fail).
  4. Honest limitations.
  5. The next smallest step.
  6. A suggested commit message (I commit manually).
Then update HACKATHON_CONTEXT.md status log and docs/SESSION_LOG.md with the verified result.
```

---

## Session cheat-sheet (what each new session should tackle)
The agent figures this out from `docs/PLAN.md` + `docs/SESSION_LOG.md`, but for your reference:

- **Session 0** — scaffold ✅ done
- **Session 1** — one-sample slice: features + blind pause detection ✅ done
- **Session 2** — dataset manifest schema + test ONE forced aligner (fallback if it fails) ✅ done
- **Session 3** — controlled variant generators (pause / rate / pitch) + severity sweep (seeded) ✅ done
- **Session 4** — baseline detectors + strict evidence objects (+ optional LLM phrasing)
- **Session 5** — Streamlit dashboard
- **Session 6** — held-out evaluation with real metrics (P/R/F1, IoU, FP rate)
- **Session 7** — submission polish (README, docs ≤6 pages, AI disclosure, demo)

Keep the ✅ marks current here if you like, but `docs/SESSION_LOG.md` is the authority.
