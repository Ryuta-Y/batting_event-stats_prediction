# Split Contracts v1

Artifacts:

```text
manifests/splits/player_group_split_v1.parquet
manifests/splits/temporal_split_v1.parquet
```

Owner: Data Agent A.

## Player Group Split

Purpose: primary benchmark split that prevents same batter-season leakage.

Required columns:

- `event_id`
- `batter_id`
- `season`
- `batter_season_id`
- `split`: `train`, `validation`, or `test`
- `split_strategy`: `player_group_v1`
- `group_key`: usually `batter_season_id`
- `created_at`
- `schema_version`

Rules:

- A `batter_season_id` must appear in exactly one split.
- Sibling rows from the same BBE or same `same_event_group_id` must stay in the same split.
- Video availability must not be used as the sole filter that defines the event universe.

## Temporal Split

Purpose: future-aware benchmark and `past_only` prior support.

Required columns:

- `event_id`
- `batter_id`
- `season`
- `batter_season_id`
- `game_date`
- `split`: `train`, `validation`, or `test`
- `split_strategy`: `temporal_v1`
- `cutoff_date`
- `created_at`
- `schema_version`

Rules:

- `game_date` is required.
- `past_only` player-season priors may only use clips with `game_date < current_event_game_date`.
- `oracle_full_season` is analysis-only and must not be mixed into official benchmark rows.

