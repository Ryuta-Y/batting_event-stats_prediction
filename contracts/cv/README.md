# CV Contracts

Owner: Agent B CV Preprocessing.

These contracts define the lifecycle after `manifests/video_sources_v1.parquet`:

```text
video_sources_v1
  -> candidate_segments_v1
  -> clips_v1
  -> CV raw artifacts + debug_overlays_v1
```

Large video files, model weights, detections, pose output, clips, and debug overlays belong under the Drive artifact root:

```text
/content/drive/MyDrive/baseball_vision
```

Reusable code belongs under:

```text
src/sport_pipeline/
```

Do not store large CV artifacts in this repo.

