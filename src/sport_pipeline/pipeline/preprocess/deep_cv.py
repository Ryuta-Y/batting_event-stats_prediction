"""Colab-only deep CV artifact runner.

This module adds the heavy CV layer that the design expects after lightweight
clip preprocessing: YOLO detection/tracking, optional RTMPose, bat-line
extraction, plate/homography hints, and raw artifact validation. Imports for
Ultralytics, MMPose, OpenCV, and NumPy stay inside functions so local preflight
tests can import this module without downloading models.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlretrieve

from sport_pipeline.artifact_check import write_json
from sport_pipeline.cv import (
    BAT_LINES_SCHEMA,
    DETECTIONS_SCHEMA,
    HOMOGRAPHY_SCHEMA,
    OBJECTS_SCHEMA,
    POSE2D_SCHEMA,
    TRACKS_SCHEMA,
)
from sport_pipeline.io import read_table, write_table
from sport_pipeline.io.runtime_cache import cache_file
from sport_pipeline.schemas.data_manifest import validate_rows


DEFAULT_PERSON_MODEL = "yolo11m.pt"
DEFAULT_OBJECT_MODEL = "yolo11s.pt"
DEFAULT_TRACKER = "bytetrack.yaml"
DEFAULT_RTMOPOSE_MODEL = "rtmpose-l_8xb32-270e_coco-wholebody-384x288"
DEFAULT_POSE_BACKEND = "rtmpose"
DEFAULT_MEDIAPIPE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"


@dataclass(frozen=True)
class ClipRuntime:
    clip_id: str
    event_id: str
    same_event_group_id: str
    view_id: str
    clip_path: Path
    fps: float


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _clip_runtime(row: dict[str, Any], base_dir: Path) -> ClipRuntime | None:
    raw_path = row.get("clip_path")
    if _is_missing(raw_path) or not str(raw_path):
        return None
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = base_dir / path
    if not path.exists():
        return None
    return ClipRuntime(
        clip_id=str(row["clip_id"]),
        event_id=str(row["event_id"]),
        same_event_group_id=str(row["same_event_group_id"]),
        view_id=str(row["view_id"]),
        clip_path=path,
        fps=float(row.get("fps") or 30.0),
    )


def _bytes_from_mb(value: int | float | None) -> int | None:
    if value is None:
        return None
    return max(0, int(float(value) * 1024**2))


def _bytes_from_gb(value: int | float | None, default_gb: float = 20.0) -> int:
    if value is None:
        value = default_gb
    return max(0, int(float(value) * 1024**3))


def _cache_clip_runtimes(
    runtimes: list[ClipRuntime],
    *,
    cache_dir: str | Path | None,
    namespace: str,
    enabled: bool,
    num_workers: int,
    max_file_mb: float | None,
    min_free_disk_gb: float,
) -> tuple[list[ClipRuntime], dict[str, Any]]:
    stats: dict[str, Any] = {"enabled": bool(enabled and cache_dir is not None), "used": 0, "reasons": {}}
    if not enabled or cache_dir is None or not runtimes:
        return runtimes, stats
    max_file_bytes = _bytes_from_mb(max_file_mb)
    min_free_bytes = _bytes_from_gb(min_free_disk_gb)

    def stage(index: int, runtime: ClipRuntime) -> tuple[int, ClipRuntime, str, bool]:
        result = cache_file(
            runtime.clip_path,
            cache_dir=cache_dir,
            namespace=namespace,
            key=runtime.clip_id,
            enabled=True,
            max_file_bytes=max_file_bytes,
            min_free_disk_bytes=min_free_bytes,
        )
        staged = ClipRuntime(
            clip_id=runtime.clip_id,
            event_id=runtime.event_id,
            same_event_group_id=runtime.same_event_group_id,
            view_id=runtime.view_id,
            clip_path=result.path,
            fps=runtime.fps,
        )
        return index, staged, result.reason, result.used_cache

    staged_by_index: dict[int, ClipRuntime] = {}
    max_workers = max(1, int(num_workers or 1))
    if max_workers == 1:
        results = [stage(index, runtime) for index, runtime in enumerate(runtimes)]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(stage, index, runtime) for index, runtime in enumerate(runtimes)]
            for future in as_completed(futures):
                results.append(future.result())
    for index, staged, reason, used in results:
        staged_by_index[index] = staged
        stats["reasons"][reason] = int(stats["reasons"].get(reason, 0)) + 1
        if used:
            stats["used"] += 1
    return [staged_by_index.get(index, runtime) for index, runtime in enumerate(runtimes)], stats


def _model_needs_download(model_id_or_path: str) -> bool:
    model_path = Path(model_id_or_path)
    if model_path.exists():
        return False
    return "/" in model_id_or_path or model_id_or_path.endswith(".pt")


def _ensure_download_allowed(model_id_or_path: str, allow_model_download: bool, label: str) -> None:
    if _model_needs_download(model_id_or_path) and not allow_model_download:
        raise RuntimeError(
            f"{label} model '{model_id_or_path}' may require a Colab download. "
            "Re-run with --allow-model-download after selecting a GPU runtime."
        )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id) or names.get(str(class_id)) or class_id)
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _iter_yolo_boxes(result: Any, names: Any) -> Iterable[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = getattr(boxes, "xyxy", None)
    conf = getattr(boxes, "conf", None)
    cls = getattr(boxes, "cls", None)
    ids = getattr(boxes, "id", None)
    if xyxy is None or conf is None or cls is None:
        return []
    xyxy_values = xyxy.detach().cpu().tolist() if hasattr(xyxy, "detach") else xyxy.tolist()
    conf_values = conf.detach().cpu().tolist() if hasattr(conf, "detach") else conf.tolist()
    cls_values = cls.detach().cpu().tolist() if hasattr(cls, "detach") else cls.tolist()
    if ids is None:
        id_values = [None] * len(xyxy_values)
    else:
        id_values = ids.detach().cpu().tolist() if hasattr(ids, "detach") else ids.tolist()
    rows = []
    for index, (bbox, confidence, class_value, track_value) in enumerate(
        zip(xyxy_values, conf_values, cls_values, id_values)
    ):
        class_id = int(class_value)
        rows.append(
            {
                "box_index": index,
                "class_id": class_id,
                "class_name": _class_name(names, class_id),
                "confidence": float(confidence),
                "bbox": [float(item) for item in bbox],
                "track_id": None if track_value is None else int(track_value),
            }
        )
    return rows


def run_yolo_track(
    clips: Iterable[ClipRuntime],
    *,
    model_name: str,
    tracker: str,
    allow_model_download: bool,
    max_frames_per_clip: int | None = None,
    progress_callback: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run Ultralytics YOLO tracking and return detection/track/debug rows."""

    _ensure_download_allowed(model_name, allow_model_download, "YOLO")
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:  # pragma: no cover - Colab dependency path
        raise RuntimeError("Ultralytics is required. In Colab run `pip install -q ultralytics`.") from exc

    model = YOLO(model_name)
    detection_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    track_accumulator: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)

    for clip_index, clip in enumerate(clips, start=1):
        frame_count = 0
        stream = model.track(
            source=str(clip.clip_path),
            stream=True,
            persist=True,
            tracker=tracker,
            verbose=False,
        )
        for frame_index, result in enumerate(stream):
            if max_frames_per_clip is not None and frame_index >= max_frames_per_clip:
                break
            frame_count += 1
            time_sec = frame_index / clip.fps if clip.fps else 0.0
            for box in _iter_yolo_boxes(result, getattr(model, "names", None)):
                x1, y1, x2, y2 = box["bbox"]
                detection_id = f"det_{clip.clip_id}_{frame_index}_{box['box_index']}"
                row = {
                    "schema_version": DETECTIONS_SCHEMA.version,
                    "detection_id": detection_id,
                    "clip_id": clip.clip_id,
                    "event_id": clip.event_id,
                    "same_event_group_id": clip.same_event_group_id,
                    "view_id": clip.view_id,
                    "frame_index": int(frame_index),
                    "time_sec": float(time_sec),
                    "class_name": str(box["class_name"]),
                    "class_id": int(box["class_id"]),
                    "confidence": float(box["confidence"]),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "track_id": box["track_id"],
                    "model_name": model_name,
                    "model_version": "ultralytics",
                    "runtime": "colab_gpu_or_cpu",
                }
                detection_rows.append(row)
                if row["track_id"] is not None:
                    track_accumulator[(clip.clip_id, int(row["track_id"]), row["class_name"])].append(row)
        debug_rows.append({"clip_id": clip.clip_id, "frames_seen": frame_count})
        if progress_callback is not None:
            progress_callback(
                clip_index,
                clip,
                detection_rows,
                _build_track_rows(track_accumulator, tracker_name=tracker, model_name=model_name),
                debug_rows,
            )

    track_rows = _build_track_rows(track_accumulator, tracker_name=tracker, model_name=model_name)
    return detection_rows, track_rows, debug_rows


def _build_track_rows(
    track_accumulator: dict[tuple[str, int, str], list[dict[str, Any]]],
    *,
    tracker_name: str,
    model_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (clip_id, track_id, class_name), detections in sorted(track_accumulator.items()):
        detections = sorted(detections, key=lambda row: row["frame_index"])
        frames = [int(row["frame_index"]) for row in detections]
        gaps = sum(1 for prev, current in zip(frames, frames[1:]) if current - prev > 1)
        first = detections[0]
        rows.append(
            {
                "schema_version": TRACKS_SCHEMA.version,
                "track_row_id": f"trk_{clip_id}_{track_id}_{class_name}",
                "clip_id": clip_id,
                "event_id": first["event_id"],
                "same_event_group_id": first["same_event_group_id"],
                "view_id": first["view_id"],
                "track_id": track_id,
                "class_name": class_name,
                "start_frame": min(frames),
                "end_frame": max(frames),
                "n_frames": len(frames),
                "mean_confidence": sum(float(row["confidence"]) for row in detections) / len(detections),
                "fragmentation": float(gaps),
                "tracker_name": tracker_name,
                "model_name": model_name,
            }
        )
    return rows


def run_rtmpose(
    clips: Iterable[ClipRuntime],
    *,
    model_name: str,
    allow_model_download: bool,
    det_model: str | None = None,
    max_frames_per_clip: int | None = None,
    progress_callback: Any | None = None,
) -> list[dict[str, Any]]:
    """Run MMPose RTMPose inference when explicitly enabled in Colab."""

    _ensure_download_allowed(model_name, allow_model_download, "RTMPose")
    try:
        from mmpose.apis import MMPoseInferencer  # type: ignore
    except ImportError as exc:  # pragma: no cover - Colab dependency path
        raise RuntimeError(
            "MMPose is required for RTMPose. Install a matching mmcv/mmpose stack in Colab first."
        ) from exc

    inferencer = MMPoseInferencer(pose2d=model_name, det_model=det_model)
    rows: list[dict[str, Any]] = []
    for clip_index, clip in enumerate(clips, start=1):
        outputs = inferencer(str(clip.clip_path), show=False, return_vis=False)
        for frame_index, output in enumerate(outputs):
            if max_frames_per_clip is not None and frame_index >= max_frames_per_clip:
                break
            predictions = output.get("predictions", []) if isinstance(output, dict) else []
            flat_predictions: list[dict[str, Any]] = []
            for item in predictions:
                if isinstance(item, list):
                    flat_predictions.extend([pred for pred in item if isinstance(pred, dict)])
                elif isinstance(item, dict):
                    flat_predictions.append(item)
            for pose_index, pred in enumerate(flat_predictions):
                keypoints = pred.get("keypoints") or pred.get("keypoints_xy") or []
                scores = pred.get("keypoint_scores") or pred.get("keypoints_score") or []
                if hasattr(keypoints, "tolist"):
                    keypoints = keypoints.tolist()
                if hasattr(scores, "tolist"):
                    scores = scores.tolist()
                score_values = [float(value) for value in scores] if isinstance(scores, list) else []
                pose_score = sum(score_values) / len(score_values) if score_values else _safe_float(pred.get("score"), 0.0)
                rows.append(
                    {
                        "schema_version": POSE2D_SCHEMA.version,
                        "pose_id": f"pose_{clip.clip_id}_{frame_index}_{pose_index}",
                        "clip_id": clip.clip_id,
                        "event_id": clip.event_id,
                        "same_event_group_id": clip.same_event_group_id,
                        "view_id": clip.view_id,
                        "frame_index": int(frame_index),
                        "time_sec": float(frame_index / clip.fps if clip.fps else 0.0),
                        "track_id": None,
                        "keypoint_schema": "coco_wholebody",
                        "keypoints_xy": keypoints,
                        "keypoint_scores": scores,
                        "pose_score": float(pose_score),
                        "model_name": model_name,
                        "model_version": "mmpose",
                        "runtime": "colab_gpu_or_cpu",
                    }
                )
        if progress_callback is not None:
            progress_callback(clip_index, clip, rows)
    return rows


def run_mediapipe_pose(
    clips: Iterable[ClipRuntime],
    *,
    max_frames_per_clip: int | None = None,
    model_asset_path: str | Path | None = None,
    model_asset_url: str = DEFAULT_MEDIAPIPE_MODEL_URL,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.3,
    min_tracking_confidence: float = 0.3,
    progress_callback: Any | None = None,
) -> list[dict[str, Any]]:
    """Run MediaPipe Pose as a Colab Python 3.12 friendly pose backend."""

    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
    except ImportError as exc:  # pragma: no cover - Colab dependency path
        raise RuntimeError("MediaPipe is required for pose inference. In Colab run `pip install -q mediapipe`.") from exc

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        return _run_mediapipe_solutions_pose(
            clips,
            max_frames_per_clip=max_frames_per_clip,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            progress_callback=progress_callback,
            cv2=cv2,
            mp=mp,
        )

    return _run_mediapipe_tasks_pose(
        clips,
        max_frames_per_clip=max_frames_per_clip,
        model_asset_path=model_asset_path,
        model_asset_url=model_asset_url,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        progress_callback=progress_callback,
        cv2=cv2,
        mp=mp,
    )


def _run_mediapipe_solutions_pose(
    clips: Iterable[ClipRuntime],
    *,
    max_frames_per_clip: int | None,
    model_complexity: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    progress_callback: Any | None,
    cv2: Any,
    mp: Any,
) -> list[dict[str, Any]]:
    """Run the legacy MediaPipe Solutions pose API when the package exposes it."""

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=int(model_complexity),
        enable_segmentation=False,
        min_detection_confidence=float(min_detection_confidence),
        min_tracking_confidence=float(min_tracking_confidence),
    )
    rows: list[dict[str, Any]] = []
    try:
        for clip_index, clip in enumerate(clips, start=1):
            cap = cv2.VideoCapture(str(clip.clip_path))
            frame_index = 0
            try:
                while True:
                    if max_frames_per_clip is not None and frame_index >= max_frames_per_clip:
                        break
                    ok, frame = cap.read()
                    if not ok:
                        break
                    height, width = frame.shape[:2]
                    result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if result.pose_landmarks is not None:
                        keypoints: list[list[float]] = []
                        scores: list[float] = []
                        for landmark in result.pose_landmarks.landmark:
                            keypoints.append([float(landmark.x * width), float(landmark.y * height)])
                            scores.append(float(getattr(landmark, "visibility", 0.0)))
                        pose_score = sum(scores) / len(scores) if scores else 0.0
                        rows.append(
                            {
                                "schema_version": POSE2D_SCHEMA.version,
                                "pose_id": f"pose_{clip.clip_id}_{frame_index}_0",
                                "clip_id": clip.clip_id,
                                "event_id": clip.event_id,
                                "same_event_group_id": clip.same_event_group_id,
                                "view_id": clip.view_id,
                                "frame_index": int(frame_index),
                                "time_sec": float(frame_index / clip.fps if clip.fps else 0.0),
                                "track_id": None,
                                "keypoint_schema": "mediapipe_pose_33",
                                "keypoints_xy": keypoints,
                                "keypoint_scores": scores,
                                "pose_score": float(pose_score),
                                "model_name": "mediapipe_pose",
                                "model_version": str(getattr(mp, "__version__", "unknown")),
                                "runtime": "colab_cpu",
                            }
                        )
                    frame_index += 1
            finally:
                cap.release()
            if progress_callback is not None:
                progress_callback(clip_index, clip, rows)
    finally:
        pose.close()
    return rows


def _ensure_mediapipe_task_model(model_asset_path: str | Path | None, model_asset_url: str) -> Path:
    if model_asset_path is None:
        model_asset_path = Path("/content/cache/baseball_vision/models/mediapipe/pose_landmarker_lite.task")
    path = Path(model_asset_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(model_asset_url, path)
    return path


def _run_mediapipe_tasks_pose(
    clips: Iterable[ClipRuntime],
    *,
    max_frames_per_clip: int | None,
    model_asset_path: str | Path | None,
    model_asset_url: str,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    progress_callback: Any | None,
    cv2: Any,
    mp: Any,
) -> list[dict[str, Any]]:
    """Run the current MediaPipe Tasks PoseLandmarker API."""

    try:
        from mediapipe.tasks import python as mp_python  # type: ignore
        from mediapipe.tasks.python import vision  # type: ignore
    except ImportError as exc:  # pragma: no cover - Colab dependency path
        raise RuntimeError(
            "This MediaPipe build does not expose mp.solutions.pose or mediapipe.tasks.python. "
            "Install a MediaPipe version with Pose Landmarker support."
        ) from exc

    task_model = _ensure_mediapipe_task_model(model_asset_path, model_asset_url)
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(task_model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=float(min_detection_confidence),
        min_pose_presence_confidence=float(min_detection_confidence),
        min_tracking_confidence=float(min_tracking_confidence),
        output_segmentation_masks=False,
    )
    rows: list[dict[str, Any]] = []
    timestamp_ms = 0
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for clip_index, clip in enumerate(clips, start=1):
            cap = cv2.VideoCapture(str(clip.clip_path))
            frame_index = 0
            frame_step_ms = max(1, int(round(1000.0 / clip.fps))) if clip.fps else 33
            try:
                while True:
                    if max_frames_per_clip is not None and frame_index >= max_frames_per_clip:
                        break
                    ok, frame = cap.read()
                    if not ok:
                        break
                    height, width = frame.shape[:2]
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    timestamp_ms += frame_step_ms
                    if result.pose_landmarks:
                        landmarks = result.pose_landmarks[0]
                        keypoints: list[list[float]] = []
                        scores: list[float] = []
                        for landmark in landmarks:
                            keypoints.append([float(landmark.x * width), float(landmark.y * height)])
                            score = getattr(landmark, "visibility", None)
                            if score is None:
                                score = getattr(landmark, "presence", 0.0)
                            scores.append(float(score or 0.0))
                        pose_score = sum(scores) / len(scores) if scores else 0.0
                        rows.append(
                            {
                                "schema_version": POSE2D_SCHEMA.version,
                                "pose_id": f"pose_{clip.clip_id}_{frame_index}_0",
                                "clip_id": clip.clip_id,
                                "event_id": clip.event_id,
                                "same_event_group_id": clip.same_event_group_id,
                                "view_id": clip.view_id,
                                "frame_index": int(frame_index),
                                "time_sec": float(frame_index / clip.fps if clip.fps else 0.0),
                                "track_id": None,
                                "keypoint_schema": "mediapipe_pose_33",
                                "keypoints_xy": keypoints,
                                "keypoint_scores": scores,
                                "pose_score": float(pose_score),
                                "model_name": "mediapipe_pose_landmarker_lite",
                                "model_version": str(getattr(mp, "__version__", "unknown")),
                                "runtime": "colab_cpu",
                            }
                        )
                    frame_index += 1
            finally:
                cap.release()
            if progress_callback is not None:
                progress_callback(clip_index, clip, rows)
    return rows


BAT_CLASS_HINTS = ("bat", "baseball bat")
PLATE_CLASS_HINTS = ("plate", "home plate", "home_plate")
FOUL_LINE_CLASS_HINTS = ("foul", "line")


def derive_object_rows_from_detections(detection_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert bat/plate/foul-line detections into object rows."""

    object_rows: list[dict[str, Any]] = []
    for row in detection_rows:
        class_name = str(row["class_name"]).lower()
        if any(hint in class_name for hint in BAT_CLASS_HINTS):
            object_type = "bat"
        elif any(hint in class_name for hint in PLATE_CLASS_HINTS):
            object_type = "home_plate"
        elif all(hint in class_name for hint in FOUL_LINE_CLASS_HINTS):
            object_type = "foul_line"
        else:
            continue
        object_id = f"obj_{row['detection_id']}_{object_type}"
        object_rows.append(
            {
                "schema_version": OBJECTS_SCHEMA.version,
                "object_id": object_id,
                "clip_id": row["clip_id"],
                "event_id": row["event_id"],
                "same_event_group_id": row["same_event_group_id"],
                "view_id": row["view_id"],
                "frame_index": row["frame_index"],
                "time_sec": row["time_sec"],
                "object_type": object_type,
                "confidence": row["confidence"],
                "bbox_xyxy": [row["x1"], row["y1"], row["x2"], row["y2"]],
                "mask_path": None,
                "source_detection_id": row["detection_id"],
                "model_name": row["model_name"],
                "model_version": row["model_version"],
            }
        )
    return object_rows


def bat_line_from_object(row: dict[str, Any]) -> dict[str, Any] | None:
    """Build a simple bat-line proxy from a bat object bounding box."""

    if row.get("object_type") != "bat":
        return None
    x1, y1, x2, y2 = [float(value) for value in row["bbox_xyxy"]]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    if width <= 0.0 or height <= 0.0:
        return None
    if width >= height:
        knob_x, knob_y = x1, (y1 + y2) / 2.0
        tip_x, tip_y = x2, (y1 + y2) / 2.0
    else:
        knob_x, knob_y = (x1 + x2) / 2.0, y2
        tip_x, tip_y = (x1 + x2) / 2.0, y1
    length = math.hypot(tip_x - knob_x, tip_y - knob_y)
    angle = math.degrees(math.atan2(tip_y - knob_y, tip_x - knob_x))
    return {
        "schema_version": BAT_LINES_SCHEMA.version,
        "bat_line_id": f"batline_{row['object_id']}",
        "clip_id": row["clip_id"],
        "event_id": row["event_id"],
        "same_event_group_id": row["same_event_group_id"],
        "view_id": row["view_id"],
        "frame_index": row["frame_index"],
        "time_sec": row["time_sec"],
        "knob_x": float(knob_x),
        "knob_y": float(knob_y),
        "tip_x": float(tip_x),
        "tip_y": float(tip_y),
        "bat_angle_deg": float(angle),
        "bat_visible_length_px": float(length),
        "confidence": float(row["confidence"]),
        "source_object_id": row["object_id"],
        "method": "bbox_long_axis_proxy",
    }


def homography_from_plate_object(row: dict[str, Any]) -> dict[str, Any] | None:
    """Build a conservative plate-rectangle homography hint from one plate bbox."""

    if row.get("object_type") != "home_plate":
        return None
    x1, y1, x2, y2 = [float(value) for value in row["bbox_xyxy"]]
    if x2 <= x1 or y2 <= y1:
        return None
    source_points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    target_points = [[0.0, 0.0], [17.0, 0.0], [17.0, 17.0], [0.0, 17.0]]
    # A full image-to-field homography needs calibrated field markers. This
    # matrix is a local plate-normalized hint, explicitly marked by method.
    matrix = [[17.0 / (x2 - x1), 0.0, -17.0 * x1 / (x2 - x1)], [0.0, 17.0 / (y2 - y1), -17.0 * y1 / (y2 - y1)], [0.0, 0.0, 1.0]]
    return {
        "schema_version": HOMOGRAPHY_SCHEMA.version,
        "homography_id": f"homo_{row['object_id']}",
        "clip_id": row["clip_id"],
        "event_id": row["event_id"],
        "same_event_group_id": row["same_event_group_id"],
        "view_id": row["view_id"],
        "frame_index": row["frame_index"],
        "time_sec": row["time_sec"],
        "homography_matrix": matrix,
        "source_points": source_points,
        "target_points": target_points,
        "confidence": float(row["confidence"]),
        "method": "plate_bbox_normalized_hint",
        "valid": True,
    }


def run_deep_cv_artifacts(
    base_dir: str | Path,
    run_id: str,
    *,
    clips_path: str | Path | None = None,
    enable_yolo: bool = False,
    enable_rtmpose: bool = False,
    yolo_model: str = DEFAULT_PERSON_MODEL,
    object_model: str | None = None,
    tracker: str = DEFAULT_TRACKER,
    pose_backend: str = DEFAULT_POSE_BACKEND,
    rtmpose_model: str = DEFAULT_RTMOPOSE_MODEL,
    rtmpose_det_model: str | None = None,
    mediapipe_model_asset_path: str | Path | None = None,
    mediapipe_model_asset_url: str = DEFAULT_MEDIAPIPE_MODEL_URL,
    mediapipe_model_complexity: int = 1,
    mediapipe_min_detection_confidence: float = 0.3,
    mediapipe_min_tracking_confidence: float = 0.3,
    allow_model_download: bool = False,
    max_clips: int | None = None,
    max_frames_per_clip: int | None = None,
    require_non_empty_detections: bool = False,
    output_suffix: str = ".parquet",
    resume: bool = True,
    checkpoint_every_clips: int = 1,
    cache_dir: str | Path | None = None,
    cache_inputs: bool = False,
    cache_num_workers: int = 4,
    cache_min_free_disk_gb: float = 20.0,
    cache_max_file_mb: float | None = None,
) -> dict[str, Path]:
    """Write raw CV artifacts. Disabled heavy steps produce valid empty rows."""

    base = Path(base_dir)
    clips_file = Path(clips_path) if clips_path else base / f"clips/{run_id}/clips_v1.parquet"
    outputs = {
        "detections": base / f"detections/{run_id}/detections_v1{output_suffix}",
        "tracks": base / f"tracks/{run_id}/tracks_v1{output_suffix}",
        "pose2d": base / f"pose2d/{run_id}/pose2d_v1{output_suffix}",
        "objects": base / f"objects/{run_id}/bat_detection_v1{output_suffix}",
        "bat_lines": base / f"objects/{run_id}/bat_line_v1{output_suffix}",
        "homography": base / f"homography/{run_id}/homography_v1{output_suffix}",
        "summary": base / f"reports/preflight/deep_cv_{run_id}.json",
        "progress": base / f"reports/preflight/deep_cv_{run_id}_progress.json",
    }
    clip_rows = read_table(clips_file) if clips_file.exists() else []
    runtimes = [runtime for row in clip_rows if (runtime := _clip_runtime(row, base)) is not None]
    if max_clips is not None:
        runtimes = runtimes[:max_clips]
    runtimes, cache_stats = _cache_clip_runtimes(
        runtimes,
        cache_dir=cache_dir,
        namespace=f"runtime_io/deep_cv/{run_id}/clips",
        enabled=cache_inputs,
        num_workers=cache_num_workers,
        max_file_mb=cache_max_file_mb,
        min_free_disk_gb=cache_min_free_disk_gb,
    )

    detection_rows: list[dict[str, Any]] = read_table(outputs["detections"]) if resume and outputs["detections"].exists() else []
    track_rows: list[dict[str, Any]] = read_table(outputs["tracks"]) if resume and outputs["tracks"].exists() else []
    pose_rows: list[dict[str, Any]] = read_table(outputs["pose2d"]) if resume and outputs["pose2d"].exists() else []
    dependency_notes: list[str] = []
    progress_payload = {}
    if resume and outputs["progress"].exists():
        try:
            progress_payload = json.loads(outputs["progress"].read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            progress_payload = {}
    completed_yolo_clip_ids = set(progress_payload.get("completed_yolo_clip_ids") or [])
    completed_object_clip_ids = set(progress_payload.get("completed_object_clip_ids") or [])
    completed_pose_clip_ids = set(progress_payload.get("completed_pose_clip_ids") or [])
    pose_backend_normalized = str(pose_backend or DEFAULT_POSE_BACKEND).lower()
    if enable_rtmpose and not pose_rows:
        completed_pose_clip_ids = set()

    def current_object_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        objects = derive_object_rows_from_detections(detection_rows)
        bat_lines = [row for row in (bat_line_from_object(obj) for obj in objects) if row is not None]
        homographies = [row for row in (homography_from_plate_object(obj) for obj in objects) if row is not None]
        return objects, bat_lines, homographies

    def write_progress(status: str) -> None:
        objects, bat_lines, homographies = current_object_rows()
        payload = {
            "schema_version": "deep_cv_progress_v1",
            "status": status,
            "run_id": run_id,
            "resume": resume,
            "clips_path": str(clips_file),
            "input_clip_rows": len(clip_rows),
            "resolved_clip_files": len(runtimes),
            "completed_yolo_clips": len(completed_yolo_clip_ids),
            "completed_object_clips": len(completed_object_clip_ids),
            "completed_pose_clips": len(completed_pose_clip_ids),
            "completed_yolo_clip_ids": sorted(completed_yolo_clip_ids),
            "completed_object_clip_ids": sorted(completed_object_clip_ids),
            "completed_pose_clip_ids": sorted(completed_pose_clip_ids),
            "pose_backend": pose_backend_normalized,
            "cache_stats": cache_stats,
            "rows": {
                "detections": len(detection_rows),
                "tracks": len(track_rows),
                "pose2d": len(pose_rows),
                "objects": len(objects),
                "bat_lines": len(bat_lines),
                "homography": len(homographies),
            },
            "outputs": {key: str(path) for key, path in outputs.items()},
        }
        outputs["progress"].parent.mkdir(parents=True, exist_ok=True)
        outputs["progress"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def checkpoint(status: str) -> None:
        objects, bat_lines, homographies = current_object_rows()
        should_write_tables = bool(detection_rows or track_rows or pose_rows or objects or not require_non_empty_detections)
        if should_write_tables:
            write_table(outputs["detections"], detection_rows)
            write_table(outputs["tracks"], track_rows)
            write_table(outputs["pose2d"], pose_rows)
            write_table(outputs["objects"], objects)
            write_table(outputs["bat_lines"], bat_lines)
            write_table(outputs["homography"], homographies)
        write_progress(status)

    if enable_yolo and runtimes:
        yolo_runtimes = [runtime for runtime in runtimes if runtime.clip_id not in completed_yolo_clip_ids]
        base_detection_rows = list(detection_rows)
        base_track_rows = list(track_rows)

        def yolo_progress(index: int, clip: ClipRuntime, detections: list[dict[str, Any]], tracks: list[dict[str, Any]], _debug: list[dict[str, Any]]) -> None:
            nonlocal detection_rows, track_rows
            detection_rows = base_detection_rows + detections
            track_rows = base_track_rows + tracks
            completed_yolo_clip_ids.add(clip.clip_id)
            if index % max(1, checkpoint_every_clips) == 0:
                checkpoint("running_yolo")

        new_detections, new_tracks, _debug = run_yolo_track(
            yolo_runtimes,
            model_name=yolo_model,
            tracker=tracker,
            allow_model_download=allow_model_download,
            max_frames_per_clip=max_frames_per_clip,
            progress_callback=yolo_progress,
        )
        if yolo_runtimes:
            detection_rows = base_detection_rows + new_detections
            track_rows = base_track_rows + new_tracks
            checkpoint("yolo_complete")
    elif not enable_yolo:
        dependency_notes.append("YOLO disabled; detections/tracks are empty contract artifacts.")

    if object_model and runtimes:
        object_runtimes = [runtime for runtime in runtimes if runtime.clip_id not in completed_object_clip_ids]

        object_detections, object_tracks, _debug = run_yolo_track(
            object_runtimes,
            model_name=object_model,
            tracker=tracker,
            allow_model_download=allow_model_download,
            max_frames_per_clip=max_frames_per_clip,
            progress_callback=None,
        )
        detection_rows.extend(object_detections)
        track_rows.extend(object_tracks)
        for runtime in object_runtimes:
            completed_object_clip_ids.add(runtime.clip_id)
        if object_runtimes:
            checkpoint("object_model_complete")
    elif not object_model:
        dependency_notes.append("No object model supplied; bat/plate rows are derived only if detector class names include them.")

    if enable_rtmpose and runtimes:
        pose_runtimes = [runtime for runtime in runtimes if runtime.clip_id not in completed_pose_clip_ids]
        base_pose_rows = list(pose_rows)

        def pose_progress(index: int, clip: ClipRuntime, poses: list[dict[str, Any]]) -> None:
            nonlocal pose_rows
            pose_rows = base_pose_rows + poses
            completed_pose_clip_ids.add(clip.clip_id)
            if index % max(1, checkpoint_every_clips) == 0:
                checkpoint("running_pose")

        if pose_backend_normalized == "mediapipe":
            if mediapipe_model_asset_path is None and cache_dir is not None:
                mediapipe_model_asset_path = Path(cache_dir) / "models/mediapipe/pose_landmarker_lite.task"
            new_pose_rows = run_mediapipe_pose(
                pose_runtimes,
                max_frames_per_clip=max_frames_per_clip,
                model_asset_path=mediapipe_model_asset_path,
                model_asset_url=mediapipe_model_asset_url,
                model_complexity=mediapipe_model_complexity,
                min_detection_confidence=mediapipe_min_detection_confidence,
                min_tracking_confidence=mediapipe_min_tracking_confidence,
                progress_callback=pose_progress,
            )
        elif pose_backend_normalized == "rtmpose":
            new_pose_rows = run_rtmpose(
                pose_runtimes,
                model_name=rtmpose_model,
                allow_model_download=allow_model_download,
                det_model=rtmpose_det_model,
                max_frames_per_clip=max_frames_per_clip,
                progress_callback=pose_progress,
            )
        else:
            raise ValueError(f"Unsupported pose_backend: {pose_backend}")
        if pose_runtimes:
            pose_rows = base_pose_rows + new_pose_rows
            checkpoint("pose_complete")
    elif not enable_rtmpose:
        dependency_notes.append("RTMPose disabled; pose2d is an empty contract artifact.")

    object_rows = derive_object_rows_from_detections(detection_rows)
    bat_line_rows = [row for row in (bat_line_from_object(obj) for obj in object_rows) if row is not None]
    homography_rows = [row for row in (homography_from_plate_object(obj) for obj in object_rows) if row is not None]

    summary_payload = {
        "schema_version": "deep_cv_summary_v1",
        "run_id": run_id,
        "clips_path": str(clips_file),
        "input_clip_rows": len(clip_rows),
        "resolved_clip_files": len(runtimes),
        "enable_yolo": enable_yolo,
        "enable_rtmpose": enable_rtmpose,
        "pose_backend": pose_backend_normalized,
        "yolo_model": yolo_model,
        "object_model": object_model,
        "tracker": tracker,
        "rtmpose_model": rtmpose_model,
        "rtmpose_det_model": rtmpose_det_model,
        "mediapipe_model_asset_path": str(mediapipe_model_asset_path) if mediapipe_model_asset_path is not None else None,
        "mediapipe_model_asset_url": mediapipe_model_asset_url,
        "mediapipe_model_complexity": mediapipe_model_complexity,
        "allow_model_download": allow_model_download,
        "max_clips": max_clips,
        "max_frames_per_clip": max_frames_per_clip,
        "require_non_empty_detections": require_non_empty_detections,
        "resume": resume,
        "checkpoint_every_clips": checkpoint_every_clips,
        "cache_dir": None if cache_dir is None else str(cache_dir),
        "cache_inputs": cache_inputs,
        "cache_num_workers": cache_num_workers,
        "cache_min_free_disk_gb": cache_min_free_disk_gb,
        "cache_max_file_mb": cache_max_file_mb,
        "cache_stats": cache_stats,
        "progress_path": str(outputs["progress"]),
        "rows": {
            "detections": len(detection_rows),
            "tracks": len(track_rows),
            "pose2d": len(pose_rows),
            "objects": len(object_rows),
            "bat_lines": len(bat_line_rows),
            "homography": len(homography_rows),
        },
        "notes": dependency_notes,
        "outputs": {key: str(path) for key, path in outputs.items() if key != "summary"},
    }
    if require_non_empty_detections and not detection_rows:
        write_json(summary_payload, outputs["summary"])
        raise RuntimeError(
            "deep CV produced 0 detections; not writing empty deep-CV artifacts in real-run mode. "
            "Enable YOLO, install ultralytics, allow model download if needed, and verify clips_v1 has real clip_path files. "
            f"summary_path={outputs['summary']}"
        )

    validate_rows(DETECTIONS_SCHEMA, detection_rows)
    validate_rows(TRACKS_SCHEMA, track_rows)
    validate_rows(POSE2D_SCHEMA, pose_rows)
    validate_rows(OBJECTS_SCHEMA, object_rows)
    validate_rows(BAT_LINES_SCHEMA, bat_line_rows)
    validate_rows(HOMOGRAPHY_SCHEMA, homography_rows)

    write_table(outputs["detections"], detection_rows)
    write_table(outputs["tracks"], track_rows)
    write_table(outputs["pose2d"], pose_rows)
    write_table(outputs["objects"], object_rows)
    write_table(outputs["bat_lines"], bat_line_rows)
    write_table(outputs["homography"], homography_rows)
    write_json(summary_payload, outputs["summary"])
    checkpoint("complete")
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Colab-only deep CV artifacts.")
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/baseball_vision")
    parser.add_argument("--run-id", default="mlb_2024_2026_full_v2")
    parser.add_argument("--clips", default=None)
    parser.add_argument("--enable-yolo", action="store_true")
    parser.add_argument("--enable-rtmpose", action="store_true")
    parser.add_argument("--yolo-model", default=DEFAULT_PERSON_MODEL)
    parser.add_argument("--object-model", default=None)
    parser.add_argument("--tracker", default=DEFAULT_TRACKER)
    parser.add_argument("--pose-backend", choices=("rtmpose", "mediapipe"), default=DEFAULT_POSE_BACKEND)
    parser.add_argument("--rtmpose-model", default=DEFAULT_RTMOPOSE_MODEL)
    parser.add_argument("--rtmpose-det-model", default=None)
    parser.add_argument("--mediapipe-model-asset-path", default=None)
    parser.add_argument("--mediapipe-model-asset-url", default=DEFAULT_MEDIAPIPE_MODEL_URL)
    parser.add_argument("--mediapipe-model-complexity", type=int, default=1)
    parser.add_argument("--mediapipe-min-detection-confidence", type=float, default=0.3)
    parser.add_argument("--mediapipe-min-tracking-confidence", type=float, default=0.3)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--max-frames-per-clip", type=int, default=None)
    parser.add_argument("--require-non-empty-detections", action="store_true")
    parser.add_argument("--output-format", choices=("parquet", "jsonl", "json", "csv"), default="parquet")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--checkpoint-every-clips", type=int, default=1)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--cache-inputs", action="store_true")
    parser.add_argument("--cache-num-workers", type=int, default=4)
    parser.add_argument("--cache-min-free-disk-gb", type=float, default=20.0)
    parser.add_argument("--cache-max-file-mb", type=float, default=None)
    args = parser.parse_args(argv)
    outputs = run_deep_cv_artifacts(
        args.base_dir,
        args.run_id,
        clips_path=args.clips,
        enable_yolo=args.enable_yolo,
        enable_rtmpose=args.enable_rtmpose,
        yolo_model=args.yolo_model,
        object_model=args.object_model,
        tracker=args.tracker,
        pose_backend=args.pose_backend,
        rtmpose_model=args.rtmpose_model,
        rtmpose_det_model=args.rtmpose_det_model,
        mediapipe_model_asset_path=args.mediapipe_model_asset_path,
        mediapipe_model_asset_url=args.mediapipe_model_asset_url,
        mediapipe_model_complexity=args.mediapipe_model_complexity,
        mediapipe_min_detection_confidence=args.mediapipe_min_detection_confidence,
        mediapipe_min_tracking_confidence=args.mediapipe_min_tracking_confidence,
        allow_model_download=args.allow_model_download,
        max_clips=args.max_clips,
        max_frames_per_clip=args.max_frames_per_clip,
        require_non_empty_detections=args.require_non_empty_detections,
        output_suffix="." + args.output_format,
        resume=not args.no_resume,
        checkpoint_every_clips=args.checkpoint_every_clips,
        cache_dir=args.cache_dir,
        cache_inputs=args.cache_inputs,
        cache_num_workers=args.cache_num_workers,
        cache_min_free_disk_gb=args.cache_min_free_disk_gb,
        cache_max_file_mb=args.cache_max_file_mb,
    )
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
