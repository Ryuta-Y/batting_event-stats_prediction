# Debug Overlays Manifest v1

Artifact:

```text
debug/{run_id}/debug_overlays_v1.parquet
```

Owner: Agent B CV Preprocessing.

Purpose: index human-review artifacts for clip quality, contact alignment, and CV failure analysis. The actual overlay mp4/png/json files live in Drive under `/content/drive/MyDrive/baseball_vision/debug/{run_id}/`.

## Required Columns

- `debug_artifact_id`
- `clip_id`
- `event_id`
- `same_event_group_id`
- `artifact_kind`
- `artifact_path`
- `frame_number`
- `time_sec`
- `view_label`
- `quality_tier`
- `created_by`
- `includes_detection`
- `includes_tracking`
- `includes_pose`
- `includes_bat`
- `includes_plate`
- `includes_contact_window`
- `review_priority`

## Artifact Kinds

Allowed examples:

- `overlay_mp4`
- `contact_frame_png`
- `trim_window_png`
- `quality_json`

Overlay output should show, when available:

- batter bbox/track id
- pose keypoints
- bat line or bat mask
- plate / foul line / homography hints
- candidate contact frame
- trim window boundaries
- `clip_status`, `quality_tier`, `view_label`, and key flags

Debug artifacts are for review and report generation. They are not model labels.

