# BBE Events Manifest v1

Artifact:

```text
manifests/bbe_events_v1.parquet
```

Owner: Data Agent A.

Purpose: one Statcast Batted Ball Event per row. This is the event universe. Video evidence is attached later and must not define the population.

## Required Columns

Core identifiers and joins:

- `event_id`: stable event key derived from `game_pk`, `at_bat_number`, `pitch_number`, and `sv_id` or `play_id`.
- `game_pk`: MLB game key.
- `game_date`: event date in `YYYY-MM-DD`.
- `season`: MLB season year.
- `batter_id`: normalized batter id, sourced from Statcast `batter`.
- `pitcher_id`: normalized pitcher id, sourced from Statcast `pitcher`.
- `batter_season_id`: `{batter_id}_{season}`.
- `at_bat_number`: Statcast at-bat number when available.
- `pitch_number`: Statcast pitch number when available.
- `play_id`: optional Statcast or Savant play id. Keep null rather than inventing one.
- `same_event_group_id`: stable grouping key for all views, crops, and augmentations of this same event. Default is `event_id`.

Statcast and context:

- `player_name`
- `events`
- `description`
- `bb_type`
- `launch_speed`
- `launch_angle`
- `launch_speed_angle`
- `estimated_ba_using_speedangle`
- `estimated_woba_using_speedangle`
- `stand`
- `p_throws`
- `pitch_type`
- `release_speed`
- `plate_x`
- `plate_z`
- `zone`
- `balls`
- `strikes`
- `outs_when_up`
- `inning`
- `inning_topbot`
- `home_team`
- `away_team`
- `sv_id`

Sampling and availability:

- `is_bbe`: true for all official rows in this artifact.
- `is_home_run`: true only when `events == home_run`.
- `dataset_role`: `train_candidate`, `eval_candidate`, `smoke_test`, or `excluded`.
- `outcome_bin`: result bin used for stratified sampling.
- `ev_bin`: exit velocity bin used for stratified sampling.
- `la_bin`: launch angle bin used for stratified sampling.
- `bb_type_bin`: batted-ball type bin.
- `has_video_candidate`: whether a candidate video has been found.
- `n_video_candidates`: count of candidate videos, including rejected ones.
- `video_availability_score`: rough score for availability/provenance quality.

Target availability:

- `target_ev_available`
- `target_la_available`
- `target_hard_hit_available`
- `target_barrel_available`
- `target_xba_available`
- `target_xwoba_available`
- `target_ops_available`
- `target_ops_missing_reason`
- `label_missing_reason`

Clean cohort and review flags:

- `clean_location_cohort_v1`: true for initial easier pitch-location cohort.
- `clean_count_cohort_v1`: true for initial easier count cohort.
- `usable_for_event_model`: candidate can be used for event-level modeling once video exists.
- `quality_flags`: JSON/list string for non-destructive quality flags.
- `outlier_flags`: JSON/list string for difficult swings or unusual mechanics.
- `review_status`: `usable_primary`, `review_only`, `excluded`, or `pending`.
- `reject_reason`: null unless `review_status == excluded`.

## Target Rules

BBE-only can define:

- EV: `launch_speed`
- LA: `launch_angle`
- hard-hit: derived from `launch_speed >= 95`
- barrel: Statcast barrel label or documented derived EV/LA rule
- optional xBA: `estimated_ba_using_speedangle`
- optional xwOBA: `estimated_woba_using_speedangle`

PA-level manifest is required for:

- OPS
- OBP
- SLG
- rolling-window OPS

Do not create event-level OPS from BBE-only data. If PA-level data is absent, set `target_ops_available=false` and a clear `target_ops_missing_reason`.

## Join Policy

Primary join candidates:

```text
game_pk, game_date, batter_id, pitcher_id, at_bat_number, pitch_number, sv_id, play_id
```

Fallback order:

1. exact `game_pk + play_id`
2. exact `game_pk + at_bat_number + pitch_number`
3. exact `game_pk + batter_id + pitcher_id + inning + inning_topbot + pitch_number`
4. manual review with `match_confidence < 1.0`

Rows with uncertain video joins remain in the manifest with availability and match reasons recorded. Do not silently drop them.

