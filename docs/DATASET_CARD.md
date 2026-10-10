# DATASET CARD

Status: schema defined (Session 0). No samples collected yet. Update as data is added.

## Design principle
Quality over size. A cleanly aligned contrastive spectrum beats a huge noisy pile.
Same transcript across a baseline/variant pair; every audio modification documented.

## Flaw classes (MVP subset first, expand after evaluation works)
1. `long_pause` — abnormally long pause, or a pause inserted inside a phrase.
2. `rate_deviation` — local speech-rate change (rushed / dragged segment).
3. `monotone` — reduced pitch variability / near-monotone delivery.
Later (only after MVP evaluated): sustained low intensity, abrupt intensity change.

## Manifest schema (one row per audio file)
| field | type | notes |
|---|---|---|
| sample_id | str | stable unique id |
| pair_id | str | links baseline ↔ its variants |
| speaker_id | str | pseudonymous |
| transcript_id | str | links identical transcripts |
| transcript_text | str | |
| audio_path | str | relative to repo |
| sample_rate | int | Hz |
| baseline_or_variant | enum | `baseline` / `variant` |
| flaw_type | str | one of the classes, or `clean` |
| severity | float | 0=none … 1=obvious; how it was built |
| gt_start_s | float | ground-truth flaw start (variants) |
| gt_end_s | float | ground-truth flaw end |
| word_range | [int,int] | affected words, if applicable |
| transform_method | str | e.g. `insert_silence` |
| transform_params | json | seeded params for reproducibility |
| alignment_method | str | aligner used |
| alignment_status | str | ok / drifted / failed |
| source_url | str | baseline provenance |
| license | str | license / permission for the source |
| split | enum | `train`/`val`/`test`, assigned by speaker+transcript |
| notes | str | known limitations |

## Provenance & licensing policy (NON-NEGOTIABLE)
- Availability online ≠ redistribution right.
- Prefer public-domain / permissively licensed recordings, or team-recorded audio with consent.
- Do NOT commit non-redistributable copyrighted audio. When a source cannot be redistributed:
  publish the source link + derived annotations/features + reproduction instructions instead.
- Every source: record `source_url` and `license`. No license = not used.

## Splits (leakage prevention)
Assign splits by **speaker AND transcript**. The same transcript or speaker must never appear
in both train and test. Variants inherit the split of their baseline (`pair_id`).

## External validity
Computer-injected flaws are known ground truth for *that manipulation* only — not proof that
the signal matches natural human delivery flaws. Add a small set of natural / team-performed
examples for external-validity checks. Report this limitation honestly.
