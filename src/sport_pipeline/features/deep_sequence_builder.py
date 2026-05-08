"""Build structured T x D features from raw CV artifacts.

This is the bridge from detector/pose/bat/plate artifacts to trainable
sequence models. It preserves event boundaries and writes the same
structured_sequence_v1 contracts used by the baseline path.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from sport_pipeline.artifact_check import write_json
from sport_pipeline.cv import BAT_LINES_SCHEMA, DETECTIONS_SCHEMA, HOMOGRAPHY_SCHEMA, POSE2D_SCHEMA
from sport_pipeline.features import STRUCTURED_FRAME_FEATURES_SCHEMA, STRUCTURED_SEQUENCE_MANIFEST_SCHEMA
from sport_pipeline.features.sequence_builder import build_frame_feature_row, build_sequence_manifest_row
from sport_pipeline.io import read_table, write_table
from sport_pipeline.schemas.data_manifest import validate_rows


DEEP_FEATURE_NAMES = (
    "pose_center_x",
    "pose_center_y",
    "pose_coverage",
    "bat_angle_deg",
    "bat_visible_length_px",
    "bat_confidence",
    "plate_homography_confidence",
    "track_count",
    "person_detection_confidence",
    "contact_phase_score",
)
DEEP_FEATURE_NAMESPACE = "deep_cv_batting_kinematics_v1"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _to_float(value: Any, default: float = 0.0) -> float:
    if _is_missing(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _split_map(base_dir: Path) -> dict[str, str]:
    for relative in (
        "manifests/splits/temporal_split_v1.parquet",
        "manifests/splits/player_group_split_v1.parquet",
        "manifests/splits/temporal_split_v1.jsonl",
        "manifests/splits/player_group_split_v1.jsonl",
    ):
        path = base_dir / relative
        if path.exists():
            return {str(row["event_id"]): str(row.get("split", "unknown")) for row in read_table(path)}
    return {}


def _frame_groups(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["clip_id"]), int(row["frame_index"]))].append(row)
    return grouped


def _nearest(grouped: dict[tuple[str, int], list[dict[str, Any]]], clip_id: str, frame_index: int, radius: int = 2) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for offset in range(radius + 1):
        for index in ({frame_index} if offset == 0 else {frame_index - offset, frame_index + offset}):
            candidates.extend(grouped.get((clip_id, index), []))
        if candidates:
            return candidates
    return []


def _pose_features(rows: list[dict[str, Any]]) -> tuple[float, float, float]:
    best = max(rows, key=lambda row: _to_float(row.get("pose_score"), 0.0), default=None)
    if best is None:
        return 0.0, 0.0, 0.0
    keypoints = best.get("keypoints_xy") or []
    scores = best.get("keypoint_scores") or []
    points = []
    for idx, point in enumerate(keypoints if isinstance(keypoints, list) else []):
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        score = _to_float(scores[idx], 0.0) if isinstance(scores, list) and idx < len(scores) else 1.0
        if score >= 0.2:
            points.append((float(point[0]), float(point[1]), score))
    if not points:
        return 0.0, 0.0, 0.0
    return mean(point[0] for point in points), mean(point[1] for point in points), len(points) / max(len(keypoints), 1)


def _detection_features(rows: list[dict[str, Any]]) -> tuple[float, float]:
    person_rows = [row for row in rows if str(row.get("class_name", "")).lower() == "person"]
    if not person_rows:
        return 0.0, 0.0
    return float(len(person_rows)), max(_to_float(row.get("confidence"), 0.0) for row in person_rows)


def _bat_features(rows: list[dict[str, Any]]) -> tuple[float, float, float]:
    best = max(rows, key=lambda row: _to_float(row.get("confidence"), 0.0), default=None)
    if best is None:
        return 0.0, 0.0, 0.0
    return (
        _to_float(best.get("bat_angle_deg"), 0.0),
        _to_float(best.get("bat_visible_length_px"), 0.0),
        _to_float(best.get("confidence"), 0.0),
    )


def _homography_feature(rows: list[dict[str, Any]]) -> float:
    valid = [row for row in rows if row.get("valid")]
    if not valid:
        return 0.0
    return max(_to_float(row.get("confidence"), 0.0) for row in valid)


def _phase_for_relative_time(relative_time: float | None) -> tuple[str, float, float]:
    if relative_time is None:
        return "unknown", 0.0, 0.0
    contact_score = max(0.0, 1.0 - min(abs(relative_time) / 0.35, 1.0))
    if relative_time < -0.90:
        return "stance_load", 0.65, contact_score
    if relative_time < -0.55:
        return "stride", 0.70, contact_score
    if relative_time < -0.20:
        return "launch", 0.75, contact_score
    if relative_time < -0.04:
        return "swing", 0.80, contact_score
    if relative_time <= 0.08:
        return "contact", 0.90, contact_score
    return "follow_through", 0.70, contact_score


def _clip_score(row: dict[str, Any]) -> tuple[float, float, float, str]:
    return (
        1.0 if row.get("clip_status") == "clean_clip" else 0.0,
        1.0 if row.get("quality_tier") == "usable_primary" else 0.0,
        _to_float(row.get("contact_confidence"), 0.0) + _to_float(row.get("view_confidence"), 0.0),
        str(row.get("clip_id", "")),
    )


def _select_clips(clip_rows: Iterable[dict[str, Any]], max_clips: int | None = None) -> list[dict[str, Any]]:
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clip_rows:
        if row.get("clip_status") == "excluded":
            continue
        by_event[str(row["event_id"])].append(row)
    selected = [sorted(rows, key=_clip_score, reverse=True)[0] for rows in by_event.values()]
    selected = sorted(selected, key=lambda row: str(row["event_id"]))
    return selected[:max_clips] if max_clips is not None else selected


def _is_deep_sequence_resume_state(
    sequence_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
) -> bool:
    """Return true only for artifacts produced by this deep-CV feature builder."""

    if not sequence_rows and not frame_rows:
        return True
    expected_names = list(DEEP_FEATURE_NAMES)
    if sequence_rows and any(list(row.get("feature_names") or []) != expected_names for row in sequence_rows):
        return False
    if frame_rows and any(row.get("feature_namespace") != DEEP_FEATURE_NAMESPACE for row in frame_rows):
        return False
    return True


def build_deep_sequence_artifacts(
    base_dir: str | Path,
    run_id: str,
    *,
    clips_path: str | Path | None = None,
    bbe_events_path: str | Path | None = None,
    detections_path: str | Path | None = None,
    pose2d_path: str | Path | None = None,
    bat_lines_path: str | Path | None = None,
    homography_path: str | Path | None = None,
    frame_count: int = 32,
    max_clips: int | None = None,
    require_non_empty: bool = False,
    output_suffix: str = ".parquet",
    resume: bool = True,
    checkpoint_every_clips: int = 25,
    structured_sequence_feature_id: str = "structured_sequence_v1",
) -> dict[str, Path]:
    """Build structured sequence rows from available deep CV artifacts."""

    base = Path(base_dir)
    clips_file = Path(clips_path) if clips_path else base / f"clips/{run_id}/clips_v1.parquet"
    bbe_file = Path(bbe_events_path) if bbe_events_path else base / "manifests/bbe_events_v1.parquet"
    detections_file = Path(detections_path) if detections_path else base / f"detections/{run_id}/detections_v1.parquet"
    pose_file = Path(pose2d_path) if pose2d_path else base / f"pose2d/{run_id}/pose2d_v1.parquet"
    bat_file = Path(bat_lines_path) if bat_lines_path else base / f"objects/{run_id}/bat_line_v1.parquet"
    homography_file = Path(homography_path) if homography_path else base / f"homography/{run_id}/homography_v1.parquet"
    outputs = {
        "structured_sequence_manifest": base / f"features/{structured_sequence_feature_id}/manifest{output_suffix}",
        "structured_sequence_frames": base / f"features/{structured_sequence_feature_id}/frames{output_suffix}",
        "summary": base / f"reports/preflight/deep_sequence_features_{run_id}.json",
        "progress": base / f"reports/preflight/deep_sequence_features_{run_id}_progress.json",
    }

    clip_rows = read_table(clips_file) if clips_file.exists() else []
    event_rows = read_table(bbe_file) if bbe_file.exists() else []
    detection_rows = read_table(detections_file) if detections_file.exists() else []
    pose_rows = read_table(pose_file) if pose_file.exists() else []
    bat_rows = read_table(bat_file) if bat_file.exists() else []
    homography_rows = read_table(homography_file) if homography_file.exists() else []

    validate_rows(DETECTIONS_SCHEMA, detection_rows)
    validate_rows(POSE2D_SCHEMA, pose_rows)
    validate_rows(BAT_LINES_SCHEMA, bat_rows)
    validate_rows(HOMOGRAPHY_SCHEMA, homography_rows)

    events = {str(row["event_id"]): row for row in event_rows}
    splits = _split_map(base)
    detections_by_frame = _frame_groups(detection_rows)
    poses_by_frame = _frame_groups(pose_rows)
    bats_by_frame = _frame_groups(bat_rows)
    homography_by_frame = _frame_groups(homography_rows)

    sequence_rows: list[dict[str, Any]] = (
        read_table(outputs["structured_sequence_manifest"])
        if resume and outputs["structured_sequence_manifest"].exists()
        else []
    )
    frame_rows: list[dict[str, Any]] = (
        read_table(outputs["structured_sequence_frames"])
        if resume and outputs["structured_sequence_frames"].exists()
        else []
    )
    resume_note = None
    if resume and not _is_deep_sequence_resume_state(sequence_rows, frame_rows):
        resume_note = (
            "existing structured_sequence_v1 artifacts were not produced by deep_cv_batting_kinematics_v1; "
            "discarded them before rebuilding deep sequence features"
        )
        sequence_rows = []
        frame_rows = []
    completed_clip_ids = {str(row.get("clip_id")) for row in sequence_rows if row.get("clip_id") is not None}
    selected_clips = _select_clips(clip_rows, max_clips=max_clips)

    def write_progress(status: str, seen_clips: int) -> None:
        payload = {
            "schema_version": "deep_sequence_features_progress_v1",
            "status": status,
            "run_id": run_id,
            "structured_sequence_feature_id": structured_sequence_feature_id,
            "resume": resume,
            "resume_note": resume_note,
            "seen_clips": seen_clips,
            "selected_clips": len(selected_clips),
            "completed_clip_ids": sorted(completed_clip_ids),
            "sequence_rows": len(sequence_rows),
            "frame_rows": len(frame_rows),
            "outputs": {key: str(path) for key, path in outputs.items()},
        }
        outputs["progress"].parent.mkdir(parents=True, exist_ok=True)
        outputs["progress"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def checkpoint(status: str, seen_clips: int) -> None:
        write_table(outputs["structured_sequence_manifest"], sequence_rows)
        write_table(outputs["structured_sequence_frames"], frame_rows)
        write_progress(status, seen_clips)

    for seen_clips, clip in enumerate(selected_clips, start=1):
        if resume and str(clip.get("clip_id")) in completed_clip_ids:
            if seen_clips % max(1, checkpoint_every_clips) == 0:
                write_progress("running_cached", seen_clips)
            continue
        event = events.get(str(clip["event_id"]), {})
        enriched = dict(clip)
        enriched["game_date"] = str(event.get("game_date", clip.get("game_date", "1970-01-01")))
        sequence = build_sequence_manifest_row(
            enriched,
            n_frames=max(2, int(frame_count)),
            feature_names=DEEP_FEATURE_NAMES,
            split=splits.get(str(clip["event_id"]), "unknown"),
            sequence_path=None,
            target_available=True,
            target_missing_reason=None,
        )
        sequence_rows.append(sequence)
        start_time = _to_float(clip.get("start_time_sec"), 0.0)
        end_time = _to_float(clip.get("end_time_sec"), start_time + _to_float(clip.get("duration_sec"), 1.0))
        contact_time = None if _is_missing(clip.get("contact_time_sec")) else _to_float(clip.get("contact_time_sec"))
        for frame_index in range(max(2, int(frame_count))):
            progress = 0.0 if frame_count <= 1 else frame_index / (frame_count - 1)
            time_sec = start_time + (end_time - start_time) * progress
            source_frame = int(round(_to_float(clip.get("start_frame"), 0.0) + progress * max(_to_float(clip.get("end_frame"), 0.0) - _to_float(clip.get("start_frame"), 0.0), 0.0)))
            relative_time = None if contact_time is None else time_sec - contact_time
            phase, phase_conf, contact_score = _phase_for_relative_time(relative_time)
            pose_x, pose_y, pose_coverage = _pose_features(_nearest(poses_by_frame, str(clip["clip_id"]), source_frame))
            bat_angle, bat_length, bat_conf = _bat_features(_nearest(bats_by_frame, str(clip["clip_id"]), source_frame))
            track_count, person_conf = _detection_features(_nearest(detections_by_frame, str(clip["clip_id"]), source_frame))
            homography_conf = _homography_feature(_nearest(homography_by_frame, str(clip["clip_id"]), source_frame))
            features = [
                pose_x,
                pose_y,
                pose_coverage,
                bat_angle,
                bat_length,
                bat_conf,
                homography_conf,
                track_count,
                person_conf,
                contact_score,
            ]
            frame_rows.append(
                build_frame_feature_row(
                    sequence,
                    frame_index=frame_index,
                    time_sec=time_sec,
                    feature_values=features,
                    relative_time_to_contact_sec=relative_time,
                    phase_label=phase,
                    phase_confidence=phase_conf,
                    feature_mask=[value != 0.0 for value in features],
                    feature_namespace=DEEP_FEATURE_NAMESPACE,
                )
            )
        completed_clip_ids.add(str(clip["clip_id"]))
        if seen_clips % max(1, checkpoint_every_clips) == 0:
            checkpoint("running", seen_clips)

    validate_rows(STRUCTURED_SEQUENCE_MANIFEST_SCHEMA, sequence_rows)
    validate_rows(STRUCTURED_FRAME_FEATURES_SCHEMA, frame_rows)
    summary_payload = {
        "schema_version": "deep_sequence_features_summary_v1",
        "run_id": run_id,
        "structured_sequence_feature_id": structured_sequence_feature_id,
        "clips_path": str(clips_file),
        "require_non_empty": require_non_empty,
        "resume": resume,
        "resume_note": resume_note,
        "checkpoint_every_clips": checkpoint_every_clips,
        "progress_path": str(outputs["progress"]),
        "inputs": {
            "detections": str(detections_file),
            "pose2d": str(pose_file),
            "bat_lines": str(bat_file),
            "homography": str(homography_file),
        },
        "input_rows": {
            "clips": len(clip_rows),
            "detections": len(detection_rows),
            "pose2d": len(pose_rows),
            "bat_lines": len(bat_rows),
            "homography": len(homography_rows),
        },
        "sequence_rows": len(sequence_rows),
        "frame_rows": len(frame_rows),
        "feature_names": list(DEEP_FEATURE_NAMES),
        "note": "Empty CV artifacts are allowed for smoke, but real mechanics features require YOLO/pose/bat/plate rows.",
        "outputs": {key: str(path) for key, path in outputs.items() if key != "summary"},
    }
    write_json(summary_payload, outputs["summary"])
    if require_non_empty and not sequence_rows:
        write_progress("failed_empty", len(selected_clips))
        raise RuntimeError(
            "deep sequence builder produced 0 sequence rows; not writing empty structured_sequence_v1 artifacts "
            "in full-run mode. Check that 12 produced non-empty clips_v1 and rerun 17. "
            f"summary_path={outputs['summary']}"
        )
    write_table(outputs["structured_sequence_manifest"], sequence_rows)
    write_table(outputs["structured_sequence_frames"], frame_rows)
    write_progress("complete", len(selected_clips))
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build structured sequence features from deep CV artifacts.")
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/baseball_vision")
    parser.add_argument("--run-id", default="mlb_2024_2026_full_v1")
    parser.add_argument("--clips", default=None)
    parser.add_argument("--bbe-events", default=None)
    parser.add_argument("--detections", default=None)
    parser.add_argument("--pose2d", default=None)
    parser.add_argument("--bat-lines", default=None)
    parser.add_argument("--homography", default=None)
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--require-non-empty", action="store_true")
    parser.add_argument("--output-format", choices=("parquet", "jsonl", "json", "csv"), default="parquet")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--checkpoint-every-clips", type=int, default=25)
    parser.add_argument("--structured-sequence-feature-id", default="structured_sequence_v1")
    args = parser.parse_args(argv)
    outputs = build_deep_sequence_artifacts(
        args.base_dir,
        args.run_id,
        clips_path=args.clips,
        bbe_events_path=args.bbe_events,
        detections_path=args.detections,
        pose2d_path=args.pose2d,
        bat_lines_path=args.bat_lines,
        homography_path=args.homography,
        frame_count=args.frame_count,
        max_clips=args.max_clips,
        require_non_empty=args.require_non_empty,
        output_suffix="." + args.output_format,
        resume=not args.no_resume,
        checkpoint_every_clips=args.checkpoint_every_clips,
        structured_sequence_feature_id=args.structured_sequence_feature_id,
    )
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
