# Sequence Dataset v1

Artifact:

```text
datasets/sequence_dataset_v1/manifest.parquet
```

Purpose: event-level sequence model input without player-season prior. This is the `event-only` structured baseline.

Required columns:

- `sample_id`
- `sequence_id`
- `clip_id`
- `event_id`
- `same_event_group_id`
- `batter_id`
- `season`
- `batter_season_id`
- `game_date`
- `split`
- `sequence_path`
- `context_feature_path`
- `target_registry_version`
- `prediction_level`
- `aggregation_scope`
- `prior_mode`
- `n_prior_clips`
- `quality_tier`
- `view_label`
- `target_available`
- `label_missing_reason`

For event-only rows:

```text
prediction_level = event
aggregation_scope = current_event_only
prior_mode = none
n_prior_clips = 0
```

The model heads must be driven by `configs/targets/target_registry_v1.yaml`. OPS is not an event-level sequence head.

