# Predictions v1

Artifact:

```text
predictions/{run_id}/predictions_v1.parquet
```

Owner: Agent D1.

`predictions_v1` is long format: one row per sample and target. Adding a head adds rows, not new wide-only columns.

## Required Columns

- `run_id`
- `sample_id`
- `event_id`
- `batter_season_id`
- `prediction_level`
- `target_name`
- `y_true`
- `y_pred`
- `target_available`
- `target_source`
- `head_kind`
- `loss_name`
- `aggregation_scope`
- `prior_mode`
- `label_missing_reason`

## Recommended Columns

- `requires_pa_manifest`
- `n_prior_clips`
- `aggregation_method`
- `same_event_ensemble`
- `prediction_std`
- `split`
- `model_family`
- `config_hash`

## Level Rules

Event-level rows represent one BBE event. Different events must not be averaged into one event prediction.

Allowed event aggregation:

- same-event views
- same-event crops
- same-event augmentations

Player-season rows must use `prediction_level=player_season`.

OPS rows must not use `prediction_level=event`; they require PA-level data.

