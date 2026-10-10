# Shared coding-agent instructions

Read `HACKATHON_CONTEXT.md` before acting. It contains the authoritative project brief, full Track C requirements, design strategy, risk guardrails, deliverables, timeline caveats, and known status.

Operate as a careful engineering teammate:
1. Inspect repository state first; never assume code/features/tests exist.
2. Implement one bounded task at a time, with explicit acceptance criteria.
3. Verify all substantive claims with actual tests or mark them unverified.
4. Keep acoustic evidence and generated explanations separate; the LLM may verbalize evidence but may not invent times, values, labels, or scores.
5. Prevent data leakage across speakers/transcripts and account for speaker variation.
6. Track data provenance, licences, consent and dataset transformations.
7. Prefer a reliable, reproducible end-to-end workflow over unnecessary complexity.
8. Explain changes simply: changed files, rationale, commands, tests/results, limits, next step.
9. Do not change the chosen Track C direction or make large architectural changes without a reason and human approval.
10. Update the master context when the team makes a decision or verifies a result.
