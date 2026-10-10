# EVALUATION

Status: protocol defined (Session 0). No results yet. Never fabricate metrics —
every number here must come from `scripts/run_eval.py` on a held-out set.

## Protocol
- Split by speaker AND transcript (see `DATASET_CARD.md`). Evaluate on unseen samples only,
  never the files used to tune thresholds.
- Include clean recordings to measure false alarms.
- Include every flaw class and several severity levels.
- Compare controlled edits vs natural performances where feasible.
- Report failures. Do not cherry-pick.

## Metrics (define computation precisely in code)
- **Event-level P / R / F1** per flaw class. An event matches ground truth if predicted and
  true intervals overlap above an IoU threshold (state the threshold).
- **Temporal localization error**: absolute start error, absolute end error, and interval IoU.
- **False-positive event rate** on clean speech (events per minute of clean audio).
- **Detection rate by class and severity** — plot detection vs severity (the "sweep curve").
- **Calibration** if a defensible confidence is produced.
- **Reproducibility**: same seed/config → same metrics across repeated runs.

## Baselines to beat / compare
1. Rule-based (pause duration + local speech rate) — the honest floor.
2. Feature-distance threshold after speaker normalization.
3. Learned classifier ONLY if enough leak-free data and time exist.

## Honesty rule
A feature changing inside the affected interval is correlation, not proof of cause. Describe the
implemented evidence chain; do not claim "causal detection" beyond what the measurement supports.

## Results
_(empty — filled in Session 6 with real numbers and at least one failure case)_
