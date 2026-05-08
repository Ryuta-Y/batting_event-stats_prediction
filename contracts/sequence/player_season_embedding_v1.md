# Player-Season Embedding v1

Artifact:

```text
features/player_season_embedding_v1/manifest.parquet
```

Purpose: mechanics prior from multiple clean clips in the same `batter_season_id`.

Required columns:

- `batter_season_id`
- `batter_id`
- `season`
- `aggregator_name`
- `aggregator_version`
- `aggregation_method`
- `aggregation_scope`
- `prior_mode`
- `cutoff_event_id`
- `cutoff_date`
- `n_clips_total`
- `n_clips_used`
- `clip_ids_used`
- `source_event_ids_used`
- `uses_future_clips`
- `embedding_path`
- `embedding_values`
- `embedding_dim`
- `quality_policy`
- `split_policy`

## Prior Modes

| mode | meaning |
|---|---|
| `none` | no player-season prior |
| `same_season_train_only` | only clean clips from training split in same batter-season |
| `past_only` | only clean clips with `game_date < current_event.game_date` |
| `oracle_full_season` | analysis-only, may include future clips except the current event |

`oracle_full_season` must be reported separately because it can use future clips.

## Aggregation Methods

- `mean_pooling`
- `quality_weighted_pooling`
- `attention_pooling`

The first two are deterministic baselines. Attention pooling is an interface placeholder until learned sequence models are trained in Colab.

