# Structured Sequence v1

Artifacts:

```text
features/structured_sequence_v1/frames.parquet
features/structured_sequence_v1/manifest.parquet
```

Purpose: represent each batting clip as a `T x D` structured sequence. This is the first video-side baseline before raw video fine-tuning.

## Frame-Level Schema

Each frame row carries:

- `sequence_id`
- `clip_id`
- `event_id`
- `same_event_group_id`
- `view_id`
- `batter_id`
- `season`
- `batter_season_id`
- `frame_index`
- `time_sec`
- `relative_time_to_contact_sec`
- `phase_label`
- `phase_confidence`
- `feature_namespace`
- `feature_names`
- `feature_values`
- `feature_mask`
- `view_label`
- `quality_tier`
- `clip_status`

Feature groups:

- canonical pose keypoints
- keypoint confidence
- joint angles
- velocity / acceleration
- pelvis and trunk proxy
- hand velocity
- bat knob / tip
- bat angle and angular velocity
- bat visible length
- plate / foul line / homography geometry
- view and quality flags

## Phase Labels

Allowed initial labels:

- `stance_load`
- `stride`
- `launch`
- `swing`
- `contact`
- `follow_through`
- `unknown`

Phase/contact labels are auxiliary tasks. They are not Statcast target heads.

## Manifest Schema

The manifest stores one row per sequence with:

- `sample_id`
- `sequence_id`
- `clip_id`
- `event_id`
- `same_event_group_id`
- `batter_season_id`
- `game_date`
- `split`
- `sequence_path`
- `n_frames`
- `feature_dim`
- `feature_names`
- `phase_labels`
- `clip_status`
- `quality_tier`
- `target_available`
- `target_missing_reason`

Only `clean_clip` / `usable_primary` rows should enter the initial clean sequence dataset.

