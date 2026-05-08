"""Render human-review CV overlay videos from clip, detection, and pose artifacts."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from sport_pipeline.artifact_check import write_json
from sport_pipeline.cv.contracts import DEBUG_OVERLAYS_SCHEMA
from sport_pipeline.io import read_table, write_table


COCO_BODY_EDGES = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 5),
    (0, 6),
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if _is_missing(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if _is_missing(value):
            return default
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _resolve_path(value: Any, base_dir: Path) -> Path | None:
    if _is_missing(value) or not str(value):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = base_dir / path
    return path if path.exists() else None


def _read_optional_table(path: Path) -> list[dict[str, Any]]:
    return read_table(path) if path.exists() else []


def _group_by_clip_frame(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clip_id = row.get("clip_id")
        if clip_id is None:
            continue
        grouped[(str(clip_id), _safe_int(row.get("frame_index")))].append(row)
    return grouped


def _clip_contact_frame(row: dict[str, Any]) -> int | None:
    fps = _safe_float(row.get("fps"), 30.0)
    contact_time = row.get("contact_time_sec")
    start_time = _safe_float(row.get("start_time_sec"), 0.0)
    if not _is_missing(contact_time):
        return max(0, _safe_int((float(contact_time) - start_time) * fps))
    contact_frame = row.get("contact_frame")
    if not _is_missing(contact_frame):
        return max(0, _safe_int(contact_frame) - _safe_int(row.get("start_frame")))
    return None


def _draw_text(cv2: Any, frame: Any, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def _draw_box(
    cv2: Any,
    frame: Any,
    xyxy: list[float],
    color: tuple[int, int, int],
    label: str,
    *,
    thickness: int = 2,
) -> None:
    if len(xyxy) < 4:
        return
    x1, y1, x2, y2 = [_safe_int(value) for value in xyxy[:4]]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if label:
        _draw_text(cv2, frame, label[:48], (x1, max(16, y1 - 6)), color)


def _draw_detections(cv2: Any, frame: Any, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        class_name = str(row.get("class_name") or "det")
        color = (60, 220, 80) if "person" in class_name.lower() else (255, 180, 70)
        track = row.get("track_id")
        conf = _safe_float(row.get("confidence"), 0.0)
        label = f"{class_name} {conf:.2f}"
        if track is not None:
            label += f" id={track}"
        _draw_box(
            cv2,
            frame,
            [_safe_float(row.get("x1")), _safe_float(row.get("y1")), _safe_float(row.get("x2")), _safe_float(row.get("y2"))],
            color,
            label,
        )


def _draw_objects(cv2: Any, frame: Any, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        object_type = str(row.get("object_type") or "object")
        color = (0, 230, 255) if object_type == "bat" else (255, 220, 0)
        bbox = row.get("bbox_xyxy")
        if isinstance(bbox, list):
            _draw_box(cv2, frame, [_safe_float(value) for value in bbox], color, object_type, thickness=2)


def _draw_bat_lines(cv2: Any, frame: Any, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        p1 = (_safe_int(row.get("knob_x")), _safe_int(row.get("knob_y")))
        p2 = (_safe_int(row.get("tip_x")), _safe_int(row.get("tip_y")))
        cv2.line(frame, p1, p2, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(frame, p1, 4, (0, 255, 255), -1)
        cv2.circle(frame, p2, 4, (0, 180, 255), -1)


def _pose_points(row: dict[str, Any]) -> tuple[list[tuple[float, float]], list[float]]:
    raw_points = row.get("keypoints_xy") or []
    raw_scores = row.get("keypoint_scores") or []
    if hasattr(raw_points, "tolist"):
        raw_points = raw_points.tolist()
    if hasattr(raw_scores, "tolist"):
        raw_scores = raw_scores.tolist()
    points: list[tuple[float, float]] = []
    if isinstance(raw_points, list):
        for point in raw_points:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                points.append((_safe_float(point[0], math.nan), _safe_float(point[1], math.nan)))
    scores: list[float] = []
    if isinstance(raw_scores, list):
        scores = [_safe_float(value, 1.0) for value in raw_scores]
    if len(scores) < len(points):
        scores.extend([1.0] * (len(points) - len(scores)))
    return points, scores


def _valid_point(point: tuple[float, float], score: float, min_score: float) -> bool:
    x, y = point
    return score >= min_score and math.isfinite(x) and math.isfinite(y) and x > 0 and y > 0


def _draw_pose_rows(
    cv2: Any,
    frame: Any,
    rows: list[dict[str, Any]],
    *,
    min_score: float,
    max_poses_per_frame: int,
) -> None:
    sorted_rows = sorted(rows, key=lambda row: _safe_float(row.get("pose_score"), 0.0), reverse=True)
    for pose_index, row in enumerate(sorted_rows[:max_poses_per_frame]):
        points, scores = _pose_points(row)
        if not points:
            continue
        edge_color = (255, 80, 255) if pose_index == 0 else (180, 110, 255)
        point_color = (80, 255, 255) if pose_index == 0 else (100, 180, 255)
        for a, b in COCO_BODY_EDGES:
            if a >= len(points) or b >= len(points):
                continue
            if _valid_point(points[a], scores[a], min_score) and _valid_point(points[b], scores[b], min_score):
                cv2.line(
                    frame,
                    (_safe_int(points[a][0]), _safe_int(points[a][1])),
                    (_safe_int(points[b][0]), _safe_int(points[b][1])),
                    edge_color,
                    3,
                    cv2.LINE_AA,
                )
        for point, score in zip(points[:17], scores[:17]):
            if _valid_point(point, score, min_score):
                cv2.circle(frame, (_safe_int(point[0]), _safe_int(point[1])), 4, point_color, -1, cv2.LINE_AA)
        _draw_text(cv2, frame, f"pose {pose_index + 1} score={_safe_float(row.get('pose_score')):.2f}", (12, 84 + pose_index * 18), edge_color)


def _overlay_row(
    clip: dict[str, Any],
    output_path: Path,
    *,
    includes_detection: bool,
    includes_tracking: bool,
    includes_pose: bool,
    includes_bat: bool,
    includes_plate: bool,
) -> dict[str, Any]:
    return {
        "schema_version": DEBUG_OVERLAYS_SCHEMA.version,
        "debug_artifact_id": f"dbg_{clip['clip_id']}_cv_overlay_mp4",
        "clip_id": clip["clip_id"],
        "event_id": clip["event_id"],
        "same_event_group_id": clip["same_event_group_id"],
        "artifact_kind": "overlay_mp4",
        "artifact_path": str(output_path),
        "frame_number": clip.get("contact_frame"),
        "time_sec": clip.get("contact_time_sec"),
        "view_label": clip.get("view_label", "unknown"),
        "quality_tier": clip.get("quality_tier", "unknown"),
        "created_by": "sport_pipeline.cv.overlay_video",
        "includes_detection": includes_detection,
        "includes_tracking": includes_tracking,
        "includes_pose": includes_pose,
        "includes_bat": includes_bat,
        "includes_plate": includes_plate,
        "includes_contact_window": True,
        "review_priority": 1 if clip.get("clip_status") != "clean_clip" else 3,
    }


def _dedupe_debug_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        debug_id = str(row.get("debug_artifact_id") or "")
        if debug_id:
            by_id[debug_id] = row
    return list(by_id.values())


def _select_clips(
    clip_rows: list[dict[str, Any]],
    base_dir: Path,
    *,
    cv_clip_ids: set[str],
    max_clips: int | None,
    include_no_cv: bool,
) -> list[dict[str, Any]]:
    resolved = []
    for row in clip_rows:
        clip_path = _resolve_path(row.get("clip_path"), base_dir)
        if clip_path is None:
            continue
        clip_id = str(row.get("clip_id"))
        if not include_no_cv and clip_id not in cv_clip_ids:
            continue
        copy = dict(row)
        copy["_resolved_clip_path"] = str(clip_path)
        resolved.append(copy)

    def score(row: dict[str, Any]) -> tuple[int, int, int, str]:
        clip_id = str(row.get("clip_id"))
        has_cv = 0 if clip_id in cv_clip_ids else 1
        clean = 0 if row.get("clip_status") == "clean_clip" else 1
        usable = 0 if row.get("clean_cohort_eligible") is True else 1
        return (has_cv, clean, usable, clip_id)

    selected = sorted(resolved, key=score)
    if max_clips is not None:
        selected = selected[: max(0, int(max_clips))]
    return selected


def render_cv_overlay_videos(
    base_dir: str | Path,
    run_id: str = "mlb_2024_2026_full_v2",
    *,
    clips_path: str | Path | None = None,
    detections_path: str | Path | None = None,
    pose2d_path: str | Path | None = None,
    objects_path: str | Path | None = None,
    bat_lines_path: str | Path | None = None,
    debug_overlays_path: str | Path | None = None,
    overlay_manifest_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    progress_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_clips: int | None = 30,
    max_frames_per_clip: int | None = None,
    include_no_cv: bool = False,
    overwrite: bool = False,
    update_clips_manifest: bool = True,
    min_pose_score: float = 0.20,
    max_poses_per_frame: int = 2,
    output_suffix: str = ".parquet",
) -> dict[str, Any]:
    """Render mp4 overlays and index them in debug manifests."""

    base = Path(base_dir)
    clips_file = Path(clips_path) if clips_path else base / f"clips/{run_id}/clips_v1{output_suffix}"
    detections_file = Path(detections_path) if detections_path else base / f"detections/{run_id}/detections_v1{output_suffix}"
    pose_file = Path(pose2d_path) if pose2d_path else base / f"pose2d/{run_id}/pose2d_v1{output_suffix}"
    objects_file = Path(objects_path) if objects_path else base / f"objects/{run_id}/bat_detection_v1{output_suffix}"
    bat_lines_file = Path(bat_lines_path) if bat_lines_path else base / f"objects/{run_id}/bat_line_v1{output_suffix}"
    debug_file = Path(debug_overlays_path) if debug_overlays_path else base / f"debug/{run_id}/debug_overlays_v1{output_suffix}"
    overlay_dir = Path(output_dir) if output_dir else base / f"debug/{run_id}/cv_overlays"
    overlay_manifest = Path(overlay_manifest_path) if overlay_manifest_path else base / f"debug/{run_id}/cv_overlay_videos_v1{output_suffix}"
    resolved_summary_path = Path(summary_path) if summary_path else base / f"reports/preflight/cv_overlay_videos_{run_id}.json"
    resolved_progress_path = Path(progress_path) if progress_path else base / f"reports/preflight/cv_overlay_videos_{run_id}_progress.json"

    clip_rows = _read_optional_table(clips_file)
    detection_rows = _read_optional_table(detections_file)
    pose_rows = _read_optional_table(pose_file)
    object_rows = _read_optional_table(objects_file)
    bat_line_rows = _read_optional_table(bat_lines_file)
    debug_rows = _read_optional_table(debug_file)

    detections_by_frame = _group_by_clip_frame(detection_rows)
    poses_by_frame = _group_by_clip_frame(pose_rows)
    objects_by_frame = _group_by_clip_frame(object_rows)
    bat_lines_by_frame = _group_by_clip_frame(bat_line_rows)
    cv_clip_ids = {
        str(row.get("clip_id"))
        for row in [*detection_rows, *pose_rows, *object_rows, *bat_line_rows]
        if row.get("clip_id") is not None
    }
    selected_clips = _select_clips(clip_rows, base, cv_clip_ids=cv_clip_ids, max_clips=max_clips, include_no_cv=include_no_cv)

    def write_progress(status: str, rendered_rows: list[dict[str, Any]], errors: list[dict[str, Any]]) -> None:
        payload = {
            "schema_version": "cv_overlay_videos_progress_v1",
            "status": status,
            "run_id": run_id,
            "selected_clips": len(selected_clips),
            "rendered_overlays": len(rendered_rows),
            "errors": errors[-50:],
            "outputs": {
                "overlay_dir": str(overlay_dir),
                "overlay_manifest": str(overlay_manifest),
                "debug_overlays": str(debug_file),
                "summary": str(resolved_summary_path),
                "progress": str(resolved_progress_path),
            },
        }
        resolved_progress_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if not selected_clips:
        write_table(overlay_manifest, [])
        summary = {
            "schema_version": "cv_overlay_videos_summary_v1",
            "run_id": run_id,
            "clips_path": str(clips_file),
            "detections_path": str(detections_file),
            "pose2d_path": str(pose_file),
            "objects_path": str(objects_file),
            "bat_lines_path": str(bat_lines_file),
            "input_rows": {
                "clips": len(clip_rows),
                "detections": len(detection_rows),
                "pose2d": len(pose_rows),
                "objects": len(object_rows),
                "bat_lines": len(bat_line_rows),
            },
            "selected_clips": 0,
            "rendered_overlays": 0,
            "include_no_cv": include_no_cv,
            "max_clips": max_clips,
            "max_frames_per_clip": max_frames_per_clip,
            "overwrite": overwrite,
            "update_clips_manifest": update_clips_manifest,
            "errors": [],
            "outputs": {
                "overlay_dir": str(overlay_dir),
                "overlay_manifest": str(overlay_manifest),
                "debug_overlays": str(debug_file),
                "summary": str(resolved_summary_path),
                "progress": str(resolved_progress_path),
            },
        }
        write_json(summary, resolved_summary_path)
        write_progress("complete_no_selected_clips", [], [])
        return summary

    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on Colab/local runtime
        write_progress("failed_missing_cv2", [], [{"reason": "opencv_import_failed", "error": str(exc)}])
        raise RuntimeError("OpenCV is required to render CV overlay videos. In Colab run `pip install -q opencv-python-headless`.") from exc

    overlay_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    updated_clip_rows = [dict(row) for row in clip_rows]
    clip_index_by_id = {str(row.get("clip_id")): index for index, row in enumerate(updated_clip_rows)}

    for clip_index, clip in enumerate(selected_clips, start=1):
        clip_id = str(clip["clip_id"])
        input_path = Path(str(clip["_resolved_clip_path"]))
        output_path = overlay_dir / f"{clip_id}_cv_overlay.mp4"
        has_detection = any(key[0] == clip_id for key in detections_by_frame)
        has_pose = any(key[0] == clip_id for key in poses_by_frame)
        has_bat = any(key[0] == clip_id for key in bat_lines_by_frame) or any(
            str(row.get("clip_id")) == clip_id and str(row.get("object_type")) == "bat" for row in object_rows
        )
        has_plate = any(str(row.get("clip_id")) == clip_id and str(row.get("object_type")) == "home_plate" for row in object_rows)
        has_tracking = any(row.get("track_id") is not None for row in detection_rows if str(row.get("clip_id")) == clip_id)

        if output_path.exists() and not overwrite:
            row = _overlay_row(
                clip,
                output_path,
                includes_detection=has_detection,
                includes_tracking=has_tracking,
                includes_pose=has_pose,
                includes_bat=has_bat,
                includes_plate=has_plate,
            )
            rendered.append(row)
            if update_clips_manifest and clip_id in clip_index_by_id:
                updated_clip_rows[clip_index_by_id[clip_id]]["overlay_path"] = str(output_path)
            write_progress("running", rendered, errors)
            continue

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            errors.append({"clip_id": clip_id, "reason": "open_failed", "clip_path": str(input_path)})
            write_progress("running", rendered, errors)
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or _safe_float(clip.get("fps"), 30.0) or 30.0
        width = _safe_int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = _safe_int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            errors.append({"clip_id": clip_id, "reason": "invalid_video_shape", "clip_path": str(input_path)})
            cap.release()
            write_progress("running", rendered, errors)
            continue
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
        if not writer.isOpened():
            errors.append({"clip_id": clip_id, "reason": "writer_open_failed", "output_path": str(output_path)})
            cap.release()
            write_progress("running", rendered, errors)
            continue

        contact_frame = _clip_contact_frame(clip)
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames_per_clip is not None and frame_index >= max_frames_per_clip:
                break
            key = (clip_id, frame_index)
            _draw_detections(cv2, frame, detections_by_frame.get(key, []))
            _draw_objects(cv2, frame, objects_by_frame.get(key, []))
            _draw_bat_lines(cv2, frame, bat_lines_by_frame.get(key, []))
            _draw_pose_rows(
                cv2,
                frame,
                poses_by_frame.get(key, []),
                min_score=min_pose_score,
                max_poses_per_frame=max_poses_per_frame,
            )
            _draw_text(cv2, frame, f"{clip_id} frame={frame_index}", (12, 24), (255, 255, 255))
            _draw_text(cv2, frame, f"event={clip.get('event_id')} quality={clip.get('quality_tier')}", (12, 44), (210, 240, 255))
            _draw_text(
                cv2,
                frame,
                f"det={len(detections_by_frame.get(key, []))} pose={len(poses_by_frame.get(key, []))} bat={len(bat_lines_by_frame.get(key, []))}",
                (12, 64),
                (210, 240, 255),
            )
            if contact_frame is not None and abs(frame_index - contact_frame) <= 1:
                cv2.rectangle(frame, (3, 3), (width - 4, height - 4), (0, 0, 255), 4)
                _draw_text(cv2, frame, "CONTACT WINDOW", (12, height - 18), (0, 0, 255))
            writer.write(frame)
            frame_index += 1

        writer.release()
        cap.release()
        if not output_path.exists() or frame_index == 0:
            errors.append({"clip_id": clip_id, "reason": "empty_overlay", "output_path": str(output_path)})
            write_progress("running", rendered, errors)
            continue
        row = _overlay_row(
            clip,
            output_path,
            includes_detection=has_detection,
            includes_tracking=has_tracking,
            includes_pose=has_pose,
            includes_bat=has_bat,
            includes_plate=has_plate,
        )
        rendered.append(row)
        if update_clips_manifest and clip_id in clip_index_by_id:
            updated_clip_rows[clip_index_by_id[clip_id]]["overlay_path"] = str(output_path)
        write_progress("running", rendered, errors)

    write_table(overlay_manifest, rendered)
    if rendered:
        write_table(debug_file, _dedupe_debug_rows([*debug_rows, *rendered]))
    if update_clips_manifest and rendered:
        write_table(clips_file, updated_clip_rows)

    summary = {
        "schema_version": "cv_overlay_videos_summary_v1",
        "run_id": run_id,
        "clips_path": str(clips_file),
        "detections_path": str(detections_file),
        "pose2d_path": str(pose_file),
        "objects_path": str(objects_file),
        "bat_lines_path": str(bat_lines_file),
        "input_rows": {
            "clips": len(clip_rows),
            "detections": len(detection_rows),
            "pose2d": len(pose_rows),
            "objects": len(object_rows),
            "bat_lines": len(bat_line_rows),
        },
        "selected_clips": len(selected_clips),
        "rendered_overlays": len(rendered),
        "include_no_cv": include_no_cv,
        "max_clips": max_clips,
        "max_frames_per_clip": max_frames_per_clip,
        "overwrite": overwrite,
        "update_clips_manifest": update_clips_manifest,
        "errors": errors,
        "outputs": {
            "overlay_dir": str(overlay_dir),
            "overlay_manifest": str(overlay_manifest),
            "debug_overlays": str(debug_file),
            "summary": str(resolved_summary_path),
            "progress": str(resolved_progress_path),
        },
    }
    write_json(summary, resolved_summary_path)
    write_progress("complete", rendered, errors)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render YOLO/pose CV overlay mp4s for batting clips.")
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/baseball_vision")
    parser.add_argument("--run-id", default="mlb_2024_2026_full_v2")
    parser.add_argument("--clips-path", default=None)
    parser.add_argument("--detections-path", default=None)
    parser.add_argument("--pose2d-path", default=None)
    parser.add_argument("--objects-path", default=None)
    parser.add_argument("--bat-lines-path", default=None)
    parser.add_argument("--debug-overlays-path", default=None)
    parser.add_argument("--overlay-manifest-path", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--progress-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-clips", type=int, default=30)
    parser.add_argument("--max-frames-per-clip", type=int, default=None)
    parser.add_argument("--include-no-cv", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-update-clips-manifest", action="store_true")
    parser.add_argument("--min-pose-score", type=float, default=0.20)
    parser.add_argument("--max-poses-per-frame", type=int, default=2)
    parser.add_argument("--output-format", choices=("parquet", "jsonl", "json", "csv"), default="parquet")
    args = parser.parse_args(argv)
    suffix = ".parquet" if args.output_format == "parquet" else f".{args.output_format}"
    summary = render_cv_overlay_videos(
        args.base_dir,
        args.run_id,
        clips_path=args.clips_path,
        detections_path=args.detections_path,
        pose2d_path=args.pose2d_path,
        objects_path=args.objects_path,
        bat_lines_path=args.bat_lines_path,
        debug_overlays_path=args.debug_overlays_path,
        overlay_manifest_path=args.overlay_manifest_path,
        summary_path=args.summary_path,
        progress_path=args.progress_path,
        output_dir=args.output_dir,
        max_clips=args.max_clips,
        max_frames_per_clip=args.max_frames_per_clip,
        include_no_cv=args.include_no_cv,
        overwrite=args.overwrite,
        update_clips_manifest=not args.no_update_clips_manifest,
        min_pose_score=args.min_pose_score,
        max_poses_per_frame=args.max_poses_per_frame,
        output_suffix=suffix,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
