# DATASET CARD

Status: **83 samples** across 4 speakers (3 synthetic + 1 human) and 5 distinct transcripts.
- **Train split** (52 samples): speakers `david` (male) and `hazel` (female), transcripts tx001, tx002.
- **Held-out Test split** (26 samples): speaker `zira` (female), transcripts tx003, tx004 (strictly speaker- and transcript-disjoint).
- **Validation / External Validity split** (5 samples): natural human speaker `human_sp0307` from Lab41-SRI VOiCES, transcript tx005.

## Design principle
Quality over size. A cleanly aligned contrastive spectrum beats a huge noisy pile.
Same transcript across a baseline/variant pair; every audio modification documented.

## Flaw classes
1. `long_pause` — abnormally long pause, or a pause inserted inside a phrase.
2. `rate_deviation` — local speech-rate change (rushed / dragged segment via WSOLA).
3. `monotone` — reduced pitch variability / near-monotone delivery via Praat PSOLA pitch-flattening.
4. `filler_word` — hesitation sounds / voiced energy inserted in inter-word gaps where silence is expected.

## Corpus Breakdown
| Split | Speaker ID | Speaker Type | Transcripts | Samples | Role |
|---|---|---|---|---|---|
| `train` | `david` | Synthetic (Win SAPI) | tx001, tx002 | 26 | Rule & threshold calibration |
| `train` | `hazel` | Synthetic (Win SAPI) | tx001, tx002 | 26 | Rule & threshold calibration |
| `test` | `zira` | Synthetic (Win SAPI) | tx003, tx004 | 26 | Held-out benchmark (Macro-F1 0.79, 0 FA/min) |
| `val` | `human_sp0307` | **Natural Human Speech** | tx005 | 5 | Real-human external validity check |

## Provenance & licensing
- **Synthetic Speech**: Generated locally via Windows SAPI TTS (voices `David`, `Hazel`, `Zira`).
- **Human Speech**: Utterance `ch127535-sg0042` by speaker `sp0307` from the **Lab41-SRI VOiCES Corpus**.
  - **License**: Creative Commons Attribution 4.0 International (CC-BY 4.0).
  - **Source URL**: `https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav`
  - **Audio Profile**: 16 kHz studio-quality clean human speech, male speaker.
  - **Transcript (tx005)**: *"I had that curiosity beside me at this moment."*

## Splits & Leakage Prevention
Assign splits strictly by **speaker AND transcript**.
The same transcript or speaker never appears in both train and test:
- `train`: speakers `david`, `hazel` on transcripts `tx001`, `tx002`
- `test`: speaker `zira` on transcripts `tx003`, `tx004`
- `val`: speaker `human_sp0307` on transcript `tx005`
Zero leakage is mechanically verified by `tests/test_dataset_split.py`.

## External Validity & Real-World Generalization
Computer-injected flaws provide rigorous, mathematically exact ground truth for algorithm benchmarking.
To verify that these acoustic detectors generalize to real human vocal tract dynamics (natural breathing, formant variations, coarticulation), the human baseline `human_sp0307` and its controlled variants were evaluated with `scripts/run_eval.py --split val`:
- **Long pause**: 1/1 detected (IoU: 0.987, start error: 0.008s, end error: 0.012s)
- **Rate deviation**: 1/1 detected (IoU: 0.629)
- **Monotone**: 1/1 detected (IoU: 0.967, start error: 0.035s, end error: 0.014s)
- **Filler word**: 1/1 detected (IoU: 0.832, start error: 0.035s, end error: 0.056s)
- **Clean Human Speech False Alarms**: **0.00 false events / min** (0 false alarms on 3.4s of clean human speech)
- **ECE**: 0.130 (well-calibrated confidence)
- **Epistemic boundary**: We measure acoustic deviations from a baseline, not emotional "nervousness". Real speech exhibits natural inter-word breathing which the contrastive baseline-matching handles without tripping false alarms.
