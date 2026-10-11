# Demo Script — YouTube submission video
**Target length: ~5 minutes** (requirement: 3–10 min). English audio **and** burned-in English
subtitles. Upload **unlisted** (not private). Record screen at 1080p; speak slowly.

This is a shot-by-shot script. Every command below is real and was verified on this repo. Every
number spoken is produced by `scripts/run_eval.py` — do not round or embellish on camera.

**Before recording (setup, off-camera):**
```bash
.venv\Scripts\activate
python -m pytest tests/ -q          # confirm 41 passed
python scripts/make_dataset.py      # confirm dataset present (Windows/SAPI)
streamlit run src/speechlab/app/main.py   # leave running in a browser tab
```
Have ready: the dashboard tab, a terminal, and this repo open in an editor.

---

## Shot 1 — The problem (0:00–0:25)
**On screen:** title slide → the running dashboard.
**Say:** "Most speech-feedback tools give a vague score — 'be more confident' — but never tell
you *where* the problem is or *what measurement* proves it. We built a system that locates
delivery flaws in time, blind, and explains each one from measured acoustic evidence it cannot
fake."

## Shot 2 — The contrastive dataset (0:25–1:05)
**On screen:** `data/manifest/manifest.csv` open; scroll the rows. Show `docs/DATASET_CARD.md`.
**Say:** "The core idea is controlled pairs. We take one clean recording and make flawed copies
that keep the *exact same words* but change one thing: a long pause, a rushed segment, or
flattened pitch — at four severities from subtle to obvious. Because we authored each edit, we
know the exact ground-truth interval. 78 clips across 3 speakers and 4 transcripts, split so the
test speaker and test sentences are never seen in training. The whole dataset regenerates from
one seed — byte-for-byte identical."

## Shot 3 — Pick a clip and analyze it blind (1:05–2:00)
**On screen:** in the dashboard sidebar, select a `long_pause` severity-1.0 variant. Click
through: baseline player, candidate player (play ~3 s of each so the pause is audible), then the
loudness timeline with the shaded detected interval.
**Say:** "I'll pick a flawed clip. Here's the original and the flawed version — you can hear the
inserted pause. The detector receives audio only; it has **not** seen the answer key. On the
timeline it has shaded the interval it thinks is wrong."

## Shot 4 — The evidence object (2:00–2:45)
**On screen:** scroll to the evidence card for that detection. Point at baseline value vs
candidate value and the rule string. Point at the green no-fabrication trust badge.
**Say:** "This is the part judges should care about. The detection isn't a vibe — it's a
structured evidence object: the longest natural pause in *this speaker's* baseline, the pause we
measured in the candidate, and the exact rule that fired. A mechanical guard rejects any number
in this explanation that measurement code didn't produce. That's why the badge says zero
fabricated numbers — a language model could word this, but it physically cannot invent a
timestamp."

## Shot 5 — The reveal (2:45–3:20)
**On screen:** tick "Reveal ground truth." The dashed true-interval overlay appears next to the
shaded prediction — they nearly coincide.
**Say:** "Now the reveal. The dashed line is the true interval we injected; the shaded band is
what the blind detector predicted. They line up — for long pauses the overlap is 0.98 IoU, start
and end error about one to two hundredths of a second. That's temporal grounding, measured, not
claimed."

## Shot 6 — Held-out metrics (3:20–4:10)
**On screen:** terminal → run `python scripts/run_eval.py`; let the table print.
**Say:** "These numbers are on the held-out set — an unseen speaker saying unseen sentences,
with thresholds tuned only on the training split. Macro-F1 0.79. Every detector has perfect
precision — zero false positives — and zero false alarms per minute on clean speech. Rate and
monotone each score F1 0.86; both were completely broken before we switched to comparing each
word to how the *same speaker* said the *same word*."

## Shot 7 — Honest failure (4:10–4:40)
**On screen:** scroll to the detection-vs-severity table (the 0.25 column is all 0/2).
**Say:** "And here's what it can't do, because we won't cherry-pick. At the subtlest severity —
a half-second extra pause, a twelve-percent speed-up — all three detectors miss it. Those flaws
sit at the floor of natural variation, and the thresholds that give zero false alarms can't
separate them. We chose precision over recall, honestly, on the training data."

## Shot 8 — Close (4:40–5:00)
**On screen:** `docs/TECHNICAL.md` and the GitHub repo page.
**Say:** "So: a reproducible contrastive dataset, blind temporal grounding, evidence that can't
be faked, and honest held-out numbers including the failures. Code, data generator, and the
six-page technical doc are all in the public repo. Thanks for watching."

---

## Caption / disclosure to include in the video description
- One-line project summary + the public GitHub URL.
- "AI coding assistants (Claude Code, with review via Codex/Antigravity) assisted development.
  All metrics are produced by the code and verified by the team, not generated by an LLM."
- Dataset is synthetic (Windows SAPI voices); no third-party audio redistributed.

## Honesty guardrails for whoever records
- Do not say "detects nervousness/emotion" — we detect *acoustic deviations*, not mental states.
- Do not call the delivery-concern score a quality metric — it's a documented constant formula.
- Do not state any number not visible on screen from `run_eval.py`.
- If a live command misbehaves on camera, re-record — do not fake the output.
