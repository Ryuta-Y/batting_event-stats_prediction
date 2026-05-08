# Candidate Segments Manifest v1

Artifact:

```text
clips/{run_id}/candidate_segments_v1.parquet
```

Owner: Agent B CV Preprocessing.

Purpose: turn each candidate video source into one or more time-bounded segments that may contain the BBE swing. A row is still evidence under review, not yet a clean training clip.

## Required Join Keys

Every row must preserve:

- `event_id`
- `same_event_group_id`
- `video_source_id`
- `view_id`
- `game_pk`
- `play_id`
- `batter_id`
- `season`
- `batter_season_id`

These keys prevent clips from drifting away from the event-level Statcast target.

## Lifecycle Columns

- `candidate_segment_id`: stable segment id.
- `lifecycle_stage`: `candidate_segment`, `view_classified`, `visibility_checked`, `contact_located`, or `trimmed_clip`.
- `segment_kind`: `batting_candidate`, `replay`, `non_batting`, `dugout`, `broadcast_cutaway`, `ball_flight`, or `unknown`.
- `review_status`: `clean_clip`, `review_only`, or `excluded`.
- `status_reason`: short reason when not clean.

## Timing And View Columns

- `start_frame`, `end_frame`, `start_time_sec`, `end_time_sec`, `fps`, `duration_sec`
- `shot_change_score`
- `shot_type`
- `view_label`
- `view_confidence`
- `camera_motion_level`

Allowed `view_label` values are:

- `pitch_center_field`
- `pitch_catcher_view`
- `bat_catcher_view`
- `bat_pitcher_view`
- `bat_side`
- `broadcast_infield`
- `replay_closeup`
- `ball_flight`
- `runner_only`
- `dugout`
- `crowd`
- `graphic`
- `unknown`

## Visibility And Contact Columns

- `batter_visible`, `batter_visibility_score`
- `bat_visible`, `bat_visibility_score`
- `plate_visible`, `plate_visibility_score`
- `contact_visible`
- `contact_frame`
- `contact_time_sec`
- `contact_confidence`

## Segment Disposition

Use `review_only` for event candidates that are not primary training clips but should remain inspectable:

- replay
- broadcast cutaway
- dugout/crowd/graphic segment that may help failure review
- contact uncertainty
- extreme occlusion
- hard negative segment
- difficult pitch or severe mechanics outlier

Use `excluded` for:

- wrong event
- corrupt media
- blocked rights or inaccessible media
- no usable join evidence

Use `clean_clip` only when the segment is event-aligned, contact-visible, and passes the initial clean-cohort visibility policy.

