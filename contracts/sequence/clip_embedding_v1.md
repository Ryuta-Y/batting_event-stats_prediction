# Clip Embedding v1

Artifact:

```text
features/clip_embedding_v1/manifest.parquet
```

Purpose: one embedding per clean clip or clip variant produced by the sequence encoder. These are inputs to player-season mechanics priors and event-level sequence models.

Required columns:

- `clip_id`
- `sequence_id`
- `event_id`
- `same_event_group_id`
- `view_id`
- `batter_id`
- `season`
- `batter_season_id`
- `game_date`
- `split`
- `encoder_name`
- `encoder_version`
- `embedding_path`
- `embedding_values`
- `embedding_dim`
- `quality_tier`
- `clip_status`
- `view_label`
- `view_confidence`
- `contact_confidence`
- `pose_coverage`
- `clean_cohort_eligible`
- `target_alignment_status`

`embedding_values` is allowed in small samples/tests. Large real embeddings should use `embedding_path` under:

```text
/content/drive/MyDrive/baseball_vision/features/clip_embedding_v1/
```

Do not derive event predictions by averaging embeddings from different events. Cross-event embeddings may only feed a player-season mechanics prior.

