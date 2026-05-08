# CV Artifacts v1

Owner: Agent B CV Preprocessing.

This file names downstream CV artifacts without requiring local heavy inference.

## Inputs

```text
manifests/bbe_events_v1.parquet
manifests/video_sources_v1.parquet
```

The BBE manifest remains the event universe. Videos are evidence attached to those events.

## Primary Agent B Outputs

```text
clips/{run_id}/candidate_segments_v1.parquet
clips/{run_id}/clips_v1.parquet
debug/{run_id}/debug_overlays_v1.parquet
```

## Raw CV Outputs For Agent C

```text
detections/{run_id}/detections_v1.parquet
tracks/{run_id}/tracks_v1.parquet
pose2d/{run_id}/pose2d_v1.parquet
objects/{run_id}/bat_detection_v1.parquet
objects/{run_id}/bat_line_v1.parquet
homography/{run_id}/homography_v1.parquet
```

Each raw CV output must include:

- `clip_id`
- `event_id`
- `same_event_group_id`
- `view_id`
- frame or timestamp
- model/runtime metadata
- confidence scores

## Model Choices

Initial Colab baseline:

- player detection: YOLO11 small/medium
- tracking: ByteTrack first, BoT-SORT comparison later
- pose: RTMPose COCO-WholeBody baseline
- bat detection: YOLO segmentation when trained; heuristic/SAM refinement only as optional
- plate/line: lightweight detector + Hough/LSD geometry

Do not download or run these heavy models locally in this repo. The implementation entrypoint is:

```bash
python -m sport_pipeline.pipeline.preprocess.deep_cv \
  --base-dir /content/drive/MyDrive/baseball_vision \
  --run-id mlb_2024_2026_full_v2 \
  --enable-yolo \
  --allow-model-download \
  --max-clips 2
```

Without `--enable-yolo` / `--enable-rtmpose`, the command writes valid empty
contract artifacts for smoke validation. With the flags enabled, the heavy
downloads and GPU inference are Colab-only human-confirmed steps.
