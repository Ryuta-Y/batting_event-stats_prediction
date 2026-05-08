# Clips Manifest v1

Artifact:

```text
clips/{run_id}/clips_v1.parquet
```

Owner: Agent B CV Preprocessing.

Purpose: contact-aligned clip metadata for downstream sequence and video models. Each row is one physical clip variant, crop, or augmentation tied to exactly one BBE event.

## Identity And Grouping

Required columns:

- `clip_id`
- `candidate_segment_id`
- `video_source_id`
- `event_id`
- `same_event_group_id`
- `view_id`
- `crop_id`
- `augmentation_id`
- `game_pk`
- `play_id`
- `batter_id`
- `season`
- `batter_season_id`

`same_event_group_id` must be inherited from `bbe_events_v1` / `video_sources_v1`. Different views, crops, and augmentations of the same BBE may be ensembled later. Different BBE events must not be averaged into one event prediction.

## Clip Status

`clip_status` has exactly these meanings:

| value | meaning | downstream use |
|---|---|---|
| `clean_clip` | event-aligned, contact-visible, good visibility, no severe outlier | initial clean cohort and primary training |
| `review_only` | candidate evidence kept for inspection, hard negatives, difficult pitch, replay, cutaway, or uncertain contact | failure browser, robustness slices, relabeling queue |
| `excluded` | wrong event, corrupt/inaccessible media, no join evidence, or unusable source | not used for training/eval |

`quality_tier` is separate:

- `usable_primary`
- `usable_with_outliers`
- `review_only`
- `excluded`

This keeps initial research clean while retaining future robustness cohorts.

## Required Quality Columns

- `view_label`
- `view_confidence`
- `batter_visible`
- `batter_visibility_score`
- `bat_visible`
- `bat_visibility_score`
- `plate_visible`
- `plate_visibility_score`
- `contact_frame`
- `contact_time_sec`
- `contact_confidence`
- `quality_flags`
- `outlier_flags`

Minimum initial clean thresholds live in:

```text
configs/cv/cv_preprocess_v1.json
```

## Contact-Centered Trim Policy

The primary mechanics clip is:

```text
clip_version = pre_contact_long
pre_contact_sec = 1.20
post_contact_sec = 0.20
```

Additional allowed versions:

- `contact_window`: contact - 0.50s to contact + 0.30s
- `full_swing`: contact - 1.60s to contact + 0.60s
- `raw_source_window`: untrimmed source window for failure review

When contact is unavailable, keep the row as `review_only` and do not invent a contact-aligned clean clip.

## Initial Clean Cohort

`clean_cohort_eligible=true` requires:

- `clip_status=clean_clip`
- `quality_tier=usable_primary`
- supported batter-mechanics view
- contact visible with enough confidence
- batter, bat, and plate visibility above thresholds
- no severe outlier flags
- not a hard negative
- not a difficult pitch for the initial clean cohort

Difficult pitch, extreme posture, check-swing-like, lost-balance, replay, and cutaway examples must remain flagged instead of silently disappearing.

