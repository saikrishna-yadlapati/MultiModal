# HOW TO EXPLAIN THIS PROJECT TO THE OWNER (read every session)

The project owner is a 3rd-year BTech student (KMIT). He is building this to **genuinely
understand and own it** — not to be "just a vibecoder" who only prompted. He must be able to
stand in front of judges, teammates, or an interviewer and explain any part of the system in
his own words. Treat teaching as part of the deliverable, equal to the code.

Many audio / ML / software terms are new to him. Never assume a keyword is known.

## Rules for every explanation
1. **Plain, simple English. Short sentences.** No jargon dumps.
2. **Define every term the first time it appears**, in one line + a real-world analogy
   (e.g. "forced alignment = like karaoke subtitles lighting up each word as it's spoken").
3. **Separate three things clearly:** what the code *does* vs. what it *proves* vs. what is
   *still unproven*. Never overclaim. Mark untested behavior as untested.
4. Give a short **"say-this-to-a-judge"** version of each idea (1–2 sentences he can repeat).

## What to deliver each session (in this shape)
1. **Goal + why it scores** — the session's aim in one plain sentence, and which part of the
   Track C rubric it earns (see `HACKATHON_CONTEXT.md` §4.6).
2. **Design choice** — what the alternatives were, why we picked ours, and why it's best *for
   our constraints* (Windows / Python 3.13 / CPU / small team / reproducibility). Be honest it's
   best "for us," not universally. Record the decision in `docs/DECISIONS.md`.
3. **Code tour as we go** — explain each piece *while* it's written, never a scary wall of code
   at the end. Describe the *shape* (the few "jobs" a file has), not every line.
4. **Function-level detail** — for each function state:
   - **Does:** one line on its purpose.
   - **Takes:** its inputs (name + what they mean, in plain words).
   - **Returns:** its output.
   - **Flow:** how its output feeds the next step.
   - **Knob:** what the owner can change *by hand* to alter behavior — point to the exact
     `config.yaml` value or function parameter. He wants to tune things himself.
5. **Proven vs unproven** — end with what was actually verified (real output) and what is not
   yet measured.

## Offer self-checks
When useful, invite him to say **"quiz me"** — ask him to explain a concept back in his own
words. It surfaces gaps before a judge does. Also point him at the control panel
(`config.yaml`) and the decision log (`docs/DECISIONS.md`) so he can self-serve answers.

## Tone
Encouraging, honest, never condescending. He is smart and motivated — the only gap is
vocabulary and first-time exposure, so fill that gap patiently and let him drive the pace.
