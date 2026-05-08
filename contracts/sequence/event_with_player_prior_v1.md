# Event With Player Prior v1

Artifact:

```text
datasets/event_with_player_prior_v1/manifest.parquet
```

Purpose: event-level prediction input where the current event sequence is conditioned on a player-season mechanics prior.

This artifact must not average predictions from different events. It stores:

```text
current event clip embedding
+ player-season mechanics embedding
+ context feature pointer
-> event-level prediction heads
```

Required metadata:

- `sample_id`
- `event_id`
- `clip_id`
- `sequence_id`
- `same_event_group_id`
- `batter_id`
- `season`
- `batter_season_id`
- `game_date`
- `split`
- `context_feature_path`
- `current_clip_embedding_path`
- `current_clip_embedding_values`
- `player_season_embedding_path`
- `player_season_embedding_values`
- `prior_mode`
- `aggregation_method`
- `aggregation_scope`
- `n_prior_clips`
- `prior_clip_ids`
- `uses_future_clips`
- `prediction_level`
- `same_event_ensemble`
- target availability columns

For this artifact:

```text
prediction_level = event
aggregation_scope = current_event_with_player_season_prior
same_event_ensemble = false
```

Same-event view/crop/augmentation ensembling is a separate aggregation scope:

```text
same_event_view_crop_augmentation_ensemble
```

Do not mix same-event ensemble with player-season prior construction.

