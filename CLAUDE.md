# Project instructions for Claude Code

Before doing project work, read `HACKATHON_CONTEXT.md` in full. It is the master source of truth for the event, Track C requirements, differentiation strategy, technical guardrails, deliverables, and current status.

Also read `docs/EXPLANATION_STYLE.md` and follow it in every explanation: the owner is building this to truly understand and defend it, not to vibecode. Teach as you build — plain English, define every term, explain design choices and code (does/takes/returns/flow + tunable knobs) as you go.

Rules:
- Inspect the repository and existing files before changing anything.
- Work in small, testable steps. State the goal and acceptance test before implementation.
- Do not build the whole app in one giant change. First prove one audio sample can be analyzed end-to-end.
- Do not invent metrics, data, licenses, capabilities, timestamps, or results. Mark untested behavior as untested.
- Never let an LLM invent numerical evidence. Explanations must be grounded in measured features and structured evidence objects.
- Review for leakage between speakers/transcripts, timestamp errors, privacy, audio licensing, reproducibility, and runtime compatibility.
- After each task, report: files changed, why, how to run/test, actual test result, limitations, and next smallest step—in beginner-friendly language.
- Do not change track, architecture, dependencies, or core scope without explaining the reason and asking for approval when the change is significant.
- Keep `HACKATHON_CONTEXT.md` and its status log updated when decisions or verified results change.
