"""Colab-side lightweight full CV preprocessing entrypoint.

This is the first production-oriented preprocessing step: it consumes
downloaded or locally referenced videos, probes/segments them with lightweight
OpenCV heuristics, writes candidate segment metadata, derives `clips_v1`, and
indexes debug review artifacts. It is intentionally model-free; heavier YOLO,
pose, bat, and plate detectors can be layered on the same contracts later.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sport_pipeline.cv import CANDIDATE_SEGMENTS_SCHEMA, CLIPS_SCHEMA, DEBUG_OVERLAYS_SCHEMA, derive_clip_metadata
from sport_pipeline.cv.diagnostics import clip_quality_diagnostics
from sport_pipeline.io import read_table, write_table
from sport_pipeline.io.runtime_cache import cache_file, cache_output_path, publish_cached_file
from sport_pipeline.schemas import BBE_EVENTS_SCHEMA, VIDEO_SOURCES_SCHEMA
from sport_pipeline.schemas.data_manifest import validate_rows
from sport_pipeline.video.source_quality import video_source_skip_reason


@dataclass(frozen=True)
class VideoProbe:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float
    readable: bool
    error: str | None = None


@dataclass(frozen=True)
class ContactEstimate:
    frame: int | None
    time_sec: float | None
    confidence: float
    visible: bool
    method: str


@dataclass(frozen=True)
class SourceProcessResult:
    seen_sources: int
    candidate: dict[str, Any] | None
    clip: dict[str, Any] | None
    debug_rows: list[dict[str, Any]]
    skipped: dict[str, Any] | None
    input_cache_used: bool
    input_cache_reason: str | None
    output_cache_used: bool


def _import_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except ImportError:
        return None


def probe_video(video_path: str | Path) -> VideoProbe:
    """Probe basic video metadata with OpenCV when available."""

    cv2 = _import_cv2()
    if cv2 is None:
        return VideoProbe(30.0, 0, 0, 0, 0.0, False, "opencv-python-headless is not installed")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return VideoProbe(30.0, 0, 0, 0, 0.0, False, "cv2.VideoCapture could not open video")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    cap.release()
    return VideoProbe(fps, frame_count, width, height, duration, True)


def estimate_contact_frame(
    video_path: str | Path,
    probe: VideoProbe,
    max_scan_frames: int = 240,
    search_start_sec: float = 0.0,
    search_end_sec: float | None = None,
) -> ContactEstimate:
    """Estimate a contact-like frame from motion energy."""

    if not probe.readable or probe.frame_count <= 0:
        return ContactEstimate(None, None, 0.0, False, "unreadable_video")
    start_frame = max(0, int(search_start_sec * probe.fps))
    end_limit_sec = search_end_sec if search_end_sec is not None else probe.duration_sec
    end_frame = min(probe.frame_count, max(start_frame + 1, int(end_limit_sec * probe.fps)))
    cv2 = _import_cv2()
    if cv2 is None:
        midpoint = max((start_frame + end_frame) // 2, 0)
        return ContactEstimate(midpoint, midpoint / probe.fps if probe.fps else None, 0.25, False, "midpoint_no_cv2")
    search_frame_count = max(1, end_frame - start_frame)
    stride = max(1, search_frame_count // max(max_scan_frames, 1))
    cap = cv2.VideoCapture(str(video_path))
    prev_gray = None
    best_frame = None
    best_score = -1.0
    scores: list[float] = []
    for frame_index in range(start_frame, end_frame, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 90))
        if prev_gray is not None:
            score = float(cv2.absdiff(gray, prev_gray).mean())
            scores.append(score)
            if score > best_score:
                best_score = score
                best_frame = frame_index
        prev_gray = gray
    cap.release()
    if best_frame is None:
        midpoint = max((start_frame + end_frame) // 2, 0)
        return ContactEstimate(midpoint, midpoint / probe.fps if probe.fps else None, 0.25, False, "midpoint_no_motion_peak")
    mean_score = sum(scores) / max(len(scores), 1)
    confidence = min(0.95, max(0.30, best_score / max(mean_score * 4.0, 1e-6)))
    method = f"motion_peak_{search_start_sec:.2f}s_to_{end_limit_sec:.2f}s"
    return ContactEstimate(best_frame, best_frame / probe.fps if probe.fps else None, confidence, confidence >= 0.45, method)


def _normalize_view_label(label: str | None) -> str:
    mapping = {
        "broadcast": "broadcast_infield",
        "catcher_view": "pitch_catcher_view",
        "center_field": "pitch_center_field",
        "batter_side": "bat_side",
        "replay": "replay_closeup",
    }
    raw = str(label or "unknown")
    return mapping.get(raw, raw)


def _float_or_default(value: Any, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _has_event_level_video_match(source: dict[str, Any]) -> bool:
    """Return true when video-source evidence is stronger than same-game only."""

    reason = str(source.get("match_reason") or "")
    source_kind = str(source.get("source_kind") or "")
    match_confidence = _float_or_default(source.get("match_confidence"), 0.0)
    if source_kind == "baseball_savant_sporty" and "exact_play_id" in reason:
        return True
    evidence_markers = (
        "player_name_text_match",
        "event_text_match",
        "description_text_match",
        "exact_play_id",
    )
    return match_confidence >= 0.45 and any(marker in reason for marker in evidence_markers)


def _resolve_view_label_and_confidence(source: dict[str, Any]) -> tuple[str, float]:
    """Use source metadata to avoid treating exact-play official clips as unknown view."""

    view_label = _normalize_view_label(source.get("view_label"))
    view_conf = _float_or_default(source.get("view_confidence"), 0.0)
    if view_label != "unknown":
        return view_label, view_conf or 0.55
    if bool(source.get("is_replay")) or bool(source.get("is_non_batting_segment")):
        return view_label, view_conf
    if not _has_event_level_video_match(source):
        return view_label, view_conf
    match_confidence = _float_or_default(source.get("match_confidence"), 0.0)
    inferred_conf = max(view_conf, min(0.75, max(0.55, match_confidence)))
    return "broadcast_infield", inferred_conf


def _video_path_for_source(source: dict[str, Any], downloads_by_source: dict[str, dict[str, Any]]) -> Path | None:
    local = source.get("local_video_path")
    if local and Path(str(local)).exists():
        return Path(str(local))
    download = downloads_by_source.get(str(source["video_source_id"]))
    if download and download.get("download_status") == "downloaded":
        for key in ("planned_path", "local_video_path", "path"):
            value = download.get(key)
            if value and Path(str(value)).exists():
                return Path(str(value))
    return None


def _candidate_from_source(
    source: dict[str, Any],
    event: dict[str, Any],
    video_path: Path,
    probe: VideoProbe,
    contact: ContactEstimate,
) -> dict[str, Any]:
    view_label, view_conf = _resolve_view_label_and_confidence(source)
    batting_visibility = str(source.get("batting_visibility") or "unknown")
    batter_visible = batting_visibility in {"visible", "partial"} or view_label in {
        "bat_catcher_view",
        "pitch_catcher_view",
        "broadcast_infield",
        "bat_side",
        "bat_pitcher_view",
    }
    contact_conf = float(contact.confidence)
    duration = probe.duration_sec or max(contact.time_sec or 0.0, 1.0)
    return {
        "schema_version": CANDIDATE_SEGMENTS_SCHEMA.version,
        "candidate_segment_id": f"cseg_{source['video_source_id']}_full",
        "video_source_id": source["video_source_id"],
        "source_video_id": source.get("source_video_id"),
        "event_id": source["event_id"],
        "same_event_group_id": source["same_event_group_id"],
        "view_id": source["view_id"],
        "game_pk": int(event["game_pk"]),
        "play_id": event.get("play_id"),
        "batter_id": int(event["batter_id"]),
        "season": int(event["season"]),
        "batter_season_id": str(event["batter_season_id"]),
        "source_kind": str(source.get("source_kind") or "unknown"),
        "source_topic": str(source.get("source_topic") or "unknown"),
        "dataset_role": str(source.get("dataset_role") or event.get("dataset_role") or "train_candidate"),
        "segment_kind": "batting_candidate" if not source.get("is_non_batting_segment") else "non_batting",
        "lifecycle_stage": "contact_located" if contact.visible else "visibility_checked",
        "review_status": "clean_clip" if contact.visible and batter_visible and not source.get("is_replay") else "review_only",
        "status_reason": None if contact.visible and batter_visible else "contact_uncertain_or_visibility_low",
        "start_frame": 0,
        "end_frame": max(probe.frame_count - 1, 0),
        "start_time_sec": 0.0,
        "end_time_sec": float(duration),
        "fps": float(probe.fps),
        "duration_sec": float(duration),
        "shot_change_score": 0.0,
        "shot_type": "single_segment_heuristic",
        "view_label": view_label,
        "view_confidence": view_conf,
        "camera_motion_level": "unknown",
        "batter_visible": bool(batter_visible),
        "batter_visibility_score": 0.70 if batter_visible else 0.25,
        "bat_visible": bool(contact.visible),
        "bat_visibility_score": max(0.30, min(0.80, contact_conf)),
        "plate_visible": view_label in {"pitch_catcher_view", "bat_catcher_view", "broadcast_infield"},
        "plate_visibility_score": 0.55 if view_label in {"pitch_catcher_view", "bat_catcher_view", "broadcast_infield"} else 0.20,
        "contact_visible": bool(contact.visible),
        "contact_frame": contact.frame,
        "contact_time_sec": contact.time_sec,
        "contact_confidence": contact_conf,
        "is_replay": bool(source.get("is_replay", False)),
        "is_non_batting_segment": bool(source.get("is_non_batting_segment", False)),
        "is_dugout": False,
        "is_broadcast_cutaway": False,
        "wrong_event": False,
        "difficult_pitch": bool(event.get("zone") not in {2, 4, 5, 6, 8}) if event.get("zone") is not None else False,
        "extreme_form": False,
        "hard_negative": bool(source.get("is_non_batting_segment", False)),
        "trim_policy": "pre_contact_long",
        "trim_start_time_sec": None,
        "trim_end_time_sec": None,
        "quality_flags": [] if contact.visible else ["contact_uncertain"],
        "outlier_flags": [],
    }


def _extract_clip(video_path: Path, clip_row: dict[str, Any], output_path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = max(float(clip_row["start_time_sec"]), 0.0)
    duration = max(float(clip_row["end_time_sec"]) - start, 0.05)
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, timeout=300)
    return output_path.exists()


def _debug_row(
    *,
    clip_row: dict[str, Any],
    artifact_id_suffix: str,
    artifact_kind: str,
    artifact_path: Path,
    includes_contact_window: bool = True,
) -> dict[str, Any]:
    return {
        "schema_version": DEBUG_OVERLAYS_SCHEMA.version,
        "debug_artifact_id": f"dbg_{clip_row['clip_id']}_{artifact_id_suffix}",
        "clip_id": clip_row["clip_id"],
        "event_id": clip_row["event_id"],
        "same_event_group_id": clip_row["same_event_group_id"],
        "artifact_kind": artifact_kind,
        "artifact_path": str(artifact_path),
        "frame_number": clip_row.get("contact_frame"),
        "time_sec": clip_row.get("contact_time_sec"),
        "view_label": clip_row["view_label"],
        "quality_tier": clip_row["quality_tier"],
        "created_by": "sport_pipeline.pipeline.preprocess.full_cv",
        "includes_detection": False,
        "includes_tracking": False,
        "includes_pose": False,
        "includes_bat": False,
        "includes_plate": False,
        "includes_contact_window": includes_contact_window,
        "review_priority": 1 if clip_row["clip_status"] != "clean_clip" else 3,
    }


def _write_contact_frame_jpg(base_dir: Path, run_id: str, clip_row: dict[str, Any], source_video_path: Path) -> Path | None:
    """Write a contact-frame review image when OpenCV can read the source video."""

    cv2 = _import_cv2()
    if cv2 is None or clip_row.get("contact_frame") is None:
        return None
    output = base_dir / "debug" / run_id / "frames" / f"{clip_row['clip_id']}_contact.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source_video_path))
    if not cap.isOpened():
        cap.release()
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(clip_row["contact_frame"]))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        return None
    if cv2.imwrite(str(output), frame):
        return output
    return None


def _write_debug_artifacts(
    base_dir: Path,
    run_id: str,
    clip_row: dict[str, Any],
    source_video_path: Path,
    debug_source_video_path: Path | None = None,
) -> tuple[Path, Path | None, list[dict[str, Any]]]:
    output = base_dir / "debug" / run_id / "quality" / f"{clip_row['clip_id']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "clip_id": clip_row["clip_id"],
        "event_id": clip_row["event_id"],
        "source_video_path": str(debug_source_video_path or source_video_path),
        "clip_status": clip_row["clip_status"],
        "quality_tier": clip_row["quality_tier"],
        "view_label": clip_row["view_label"],
        "contact_frame": clip_row["contact_frame"],
        "contact_time_sec": clip_row["contact_time_sec"],
        "quality_flags": clip_row["quality_flags"],
        "outlier_flags": clip_row["outlier_flags"],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = [
        _debug_row(
            clip_row=clip_row,
            artifact_id_suffix="quality_json",
            artifact_kind="quality_json",
            artifact_path=output,
        )
    ]
    frame_path = _write_contact_frame_jpg(base_dir, run_id, clip_row, source_video_path)
    if frame_path is not None:
        rows.append(
            _debug_row(
                clip_row=clip_row,
                artifact_id_suffix="contact_frame_jpg",
                artifact_kind="contact_frame_jpg",
                artifact_path=frame_path,
            )
        )
    return output, frame_path, rows


def _bytes_from_mb(value: int | float | None) -> int | None:
    if value is None:
        return None
    return max(0, int(float(value) * 1024**2))


def _bytes_from_gb(value: int | float | None, default_gb: float = 20.0) -> int:
    if value is None:
        value = default_gb
    return max(0, int(float(value) * 1024**3))


def run_full_cv_preprocess(
    base_dir: str | Path,
    run_id: str,
    *,
    bbe_events_path: str | Path | None = None,
    video_sources_path: str | Path | None = None,
    download_manifest_path: str | Path | None = None,
    max_sources: int | None = None,
    extract_clips: bool = False,
    require_non_empty: bool = False,
    require_event_level_match: bool = False,
    min_video_match_confidence: float = 0.45,
    contact_search_start_sec: float = 0.5,
    contact_search_end_sec: float | None = 8.0,
    clip_version: str = "pre_contact_long",
    output_suffix: str = ".parquet",
    resume: bool = True,
    checkpoint_every_sources: int = 10,
    cache_dir: str | Path | None = None,
    cache_inputs: bool = False,
    cache_outputs: bool = False,
    cache_min_free_disk_gb: float = 20.0,
    cache_max_file_mb: float | None = None,
    num_workers: int = 1,
    progress_log_every_sources: int = 0,
) -> dict[str, Any]:
    """Build candidate segments, clips, and debug metadata from local videos."""

    base = Path(base_dir)
    bbe_path = Path(bbe_events_path) if bbe_events_path else base / "manifests/bbe_events_v1.parquet"
    sources_path = Path(video_sources_path) if video_sources_path else base / "manifests/video_sources_v1.parquet"
    downloads_path = (
        Path(download_manifest_path)
        if download_manifest_path
        else base / f"raw_videos/{run_id}/download_manifest_v1.parquet"
    )
    outputs = {
        "candidate_segments": base / f"clips/{run_id}/candidate_segments_v1{output_suffix}",
        "clips": base / f"clips/{run_id}/clips_v1{output_suffix}",
        "debug_overlays": base / f"debug/{run_id}/debug_overlays_v1{output_suffix}",
        "summary": base / f"reports/preflight/full_cv_preprocess_{run_id}.json",
        "progress": base / f"reports/preflight/full_cv_preprocess_{run_id}_progress.json",
    }
    effective_num_workers = max(1, int(num_workers or 1))
    # `max_sources` is a precise cap used in smoke tests and spot checks; keep
    # those runs sequential so parallel prefetching never creates extra clips.
    use_parallel = effective_num_workers > 1 and max_sources is None
    cache_root = Path(cache_dir) if cache_dir is not None and (cache_inputs or cache_outputs) else None
    cache_min_free_bytes = _bytes_from_gb(cache_min_free_disk_gb)
    cache_max_file_bytes = _bytes_from_mb(cache_max_file_mb)
    cache_policy = {
        "cache_dir": None if cache_root is None else str(cache_root),
        "cache_inputs": bool(cache_inputs and cache_root is not None),
        "cache_outputs": bool(cache_outputs and cache_root is not None),
        "cache_min_free_disk_gb": cache_min_free_disk_gb,
        "cache_max_file_mb": cache_max_file_mb,
        "num_workers": effective_num_workers,
        "parallel_enabled": use_parallel,
        "progress_log_every_sources": progress_log_every_sources,
    }
    cache_stats: dict[str, Any] = {
        "input_cache_used": 0,
        "input_cache_reasons": {},
        "output_cache_used": 0,
    }
    bbe_rows = read_table(bbe_path)
    source_rows = read_table(sources_path)
    download_rows = read_table(downloads_path) if downloads_path.exists() else []
    validate_rows(BBE_EVENTS_SCHEMA, bbe_rows)
    validate_rows(VIDEO_SOURCES_SCHEMA, source_rows)
    events_by_id = {str(row["event_id"]): row for row in bbe_rows}
    downloads_by_source = {str(row["video_source_id"]): row for row in download_rows}
    if downloads_by_source:
        source_rows_to_scan = [
            row
            for row in source_rows
            if str(row.get("video_source_id") or "") in downloads_by_source
        ]
    else:
        source_rows_to_scan = []
    skipped_not_in_download_manifest = max(0, len(source_rows) - len(source_rows_to_scan))

    candidate_rows: list[dict[str, Any]] = read_table(outputs["candidate_segments"]) if resume and outputs["candidate_segments"].exists() else []
    clip_rows: list[dict[str, Any]] = read_table(outputs["clips"]) if resume and outputs["clips"].exists() else []
    debug_rows: list[dict[str, Any]] = read_table(outputs["debug_overlays"]) if resume and outputs["debug_overlays"].exists() else []
    skipped: list[dict[str, Any]] = []
    processed = len(candidate_rows)
    completed_source_ids = {str(row.get("video_source_id")) for row in candidate_rows if row.get("video_source_id") is not None}

    def write_progress(status: str, seen_sources: int) -> None:
        diagnostics = clip_quality_diagnostics(clip_rows, limit=8)
        total_sources = len(source_rows_to_scan)
        percent = 100.0 * min(seen_sources, total_sources) / max(total_sources, 1)
        payload = {
            "schema_version": "full_cv_preprocess_progress_v1",
            "status": status,
            "run_id": run_id,
            "resume": resume,
            "seen_sources": seen_sources,
            "total_sources": total_sources,
            "progress_percent": percent,
            "source_rows_total": len(source_rows),
            "download_manifest_rows": len(download_rows),
            "skipped_not_in_download_manifest": skipped_not_in_download_manifest,
            "pending_sources": max(0, total_sources - len(completed_source_ids) - len(skipped)),
            "processed_sources": processed,
            "candidate_segments": len(candidate_rows),
            "clips": len(clip_rows),
            "clean_trainable_clips": diagnostics["clean_trainable_clips"],
            "clip_status_counts": diagnostics["clip_status_counts"],
            "quality_tier_counts": diagnostics["quality_tier_counts"],
            "review_reason_counts": diagnostics["review_reason_counts"],
            "debug_overlays": len(debug_rows),
            "skipped_count": len(skipped),
            "cache_policy": cache_policy,
            "cache_stats": cache_stats,
            "outputs": {key: str(value) for key, value in outputs.items()},
            "skipped_tail": skipped[-100:],
        }
        outputs["progress"].parent.mkdir(parents=True, exist_ok=True)
        outputs["progress"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if progress_log_every_sources > 0 and (
            seen_sources % max(1, progress_log_every_sources) == 0
            or status in {"complete", "failed_empty"}
        ):
            print(
                "full_cv clips "
                f"{len(clip_rows)}/{total_sources} "
                f"({percent:.1f}%) clean={diagnostics['clean_trainable_clips']} "
                f"processed={processed} skipped={len(skipped)} "
                f"workers={effective_num_workers} cache_in={cache_stats['input_cache_used']} "
                f"cache_out={cache_stats['output_cache_used']} status={status}"
            )

    def checkpoint(seen_sources: int, status: str = "running") -> None:
        write_table(outputs["candidate_segments"], candidate_rows)
        write_table(outputs["clips"], clip_rows)
        write_table(outputs["debug_overlays"], debug_rows)
        write_progress(status, seen_sources)

    def process_source(seen_sources: int, source: dict[str, Any]) -> SourceProcessResult:
        event = events_by_id.get(str(source["event_id"]))
        video_path = _video_path_for_source(source, downloads_by_source)
        if event is None or video_path is None:
            return SourceProcessResult(
                seen_sources,
                None,
                None,
                [],
                {
                    "video_source_id": source.get("video_source_id"),
                    "event_id": source.get("event_id"),
                    "reason": "missing_event_or_local_video",
                },
                False,
                None,
                False,
            )
        if require_event_level_match:
            skip_reason = video_source_skip_reason(source, min_match_confidence=min_video_match_confidence)
            if skip_reason is not None:
                return SourceProcessResult(
                    seen_sources,
                    None,
                    None,
                    [],
                    {
                        "video_source_id": source.get("video_source_id"),
                        "event_id": source.get("event_id"),
                        "reason": skip_reason,
                    },
                    False,
                    None,
                    False,
                )

        runtime_video_path = video_path
        input_cache_used = False
        input_cache_reason: str | None = None
        if cache_root is not None and cache_inputs:
            input_cache = cache_file(
                video_path,
                cache_dir=cache_root,
                namespace=f"runtime_io/full_cv/{run_id}/raw_videos",
                key=str(source.get("video_source_id") or ""),
                enabled=True,
                max_file_bytes=cache_max_file_bytes,
                min_free_disk_bytes=cache_min_free_bytes,
            )
            runtime_video_path = input_cache.path
            input_cache_used = input_cache.used_cache
            input_cache_reason = input_cache.reason

        probe = probe_video(runtime_video_path)
        contact = estimate_contact_frame(
            runtime_video_path,
            probe,
            search_start_sec=contact_search_start_sec,
            search_end_sec=contact_search_end_sec,
        )
        candidate = _candidate_from_source(source, event, video_path, probe, contact)
        clip = derive_clip_metadata(candidate, clip_version=clip_version)
        clip_output = base / "clips" / run_id / "videos" / f"{clip['clip_id']}.mp4"
        output_cache_used = False
        if extract_clips:
            try:
                extract_output = cache_output_path(
                    cache_root,
                    namespace=f"runtime_io/full_cv/{run_id}/clips",
                    filename=f"{clip['clip_id']}.mp4",
                    enabled=bool(cache_root is not None and cache_outputs),
                )
                if extract_output is None:
                    extract_output = clip_output
                if _extract_clip(runtime_video_path, clip, extract_output):
                    if extract_output != clip_output:
                        publish_cached_file(extract_output, clip_output)
                        output_cache_used = True
                    clip["clip_path"] = str(clip_output)
            except Exception as exc:  # pragma: no cover - ffmpeg/runtime dependent
                clip["clip_status"] = "review_only"
                clip["quality_tier"] = "review_only"
                clip["review_reason"] = f"clip_extract_failed:{exc}"
        debug_json, debug_frame, debug_artifact_rows = _write_debug_artifacts(
            base,
            run_id,
            clip,
            runtime_video_path,
            debug_source_video_path=video_path,
        )
        clip["debug_frame_path"] = str(debug_frame or debug_json)
        clip["overlay_path"] = str(debug_json)
        return SourceProcessResult(
            seen_sources,
            candidate,
            clip,
            debug_artifact_rows,
            None,
            input_cache_used,
            input_cache_reason,
            output_cache_used,
        )

    def apply_result(result: SourceProcessResult) -> None:
        nonlocal processed
        if result.input_cache_reason:
            reasons = cache_stats["input_cache_reasons"]
            reasons[result.input_cache_reason] = int(reasons.get(result.input_cache_reason, 0)) + 1
        if result.input_cache_used:
            cache_stats["input_cache_used"] += 1
        if result.output_cache_used:
            cache_stats["output_cache_used"] += 1
        if result.skipped is not None:
            skipped.append(result.skipped)
            return
        if result.candidate is None or result.clip is None:
            return
        candidate_rows.append(result.candidate)
        clip_rows.append(result.clip)
        debug_rows.extend(result.debug_rows)
        processed += 1
        completed_source_ids.add(str(result.candidate["video_source_id"]))

    def process_batch(batch: list[tuple[int, dict[str, Any]]]) -> int:
        if not batch:
            return 0
        results: list[SourceProcessResult] = []
        with ThreadPoolExecutor(max_workers=effective_num_workers) as executor:
            futures = [executor.submit(process_source, seen, source) for seen, source in batch]
            for future in as_completed(futures):
                results.append(future.result())
        max_seen = 0
        for result in sorted(results, key=lambda item: item.seen_sources):
            apply_result(result)
            max_seen = max(max_seen, result.seen_sources)
        return max_seen

    if use_parallel:
        pending_batch: list[tuple[int, dict[str, Any]]] = []
        batch_size = max(effective_num_workers, checkpoint_every_sources)
        for seen_sources, source in enumerate(source_rows_to_scan, start=1):
            if resume and str(source.get("video_source_id")) in completed_source_ids:
                if seen_sources % max(1, checkpoint_every_sources) == 0:
                    write_progress("running_cached", seen_sources)
                continue
            pending_batch.append((seen_sources, source))
            if len(pending_batch) >= batch_size:
                max_seen = process_batch(pending_batch)
                pending_batch = []
                checkpoint(max_seen, status="running_parallel")
        if pending_batch:
            max_seen = process_batch(pending_batch)
            checkpoint(max_seen, status="running_parallel")
    else:
        for seen_sources, source in enumerate(source_rows_to_scan, start=1):
            if max_sources is not None and processed >= max_sources:
                break
            if resume and str(source.get("video_source_id")) in completed_source_ids:
                if seen_sources % max(1, checkpoint_every_sources) == 0:
                    write_progress("running_cached", seen_sources)
                continue
            apply_result(process_source(seen_sources, source))
            if seen_sources % max(1, checkpoint_every_sources) == 0:
                checkpoint(seen_sources)

    diagnostics = clip_quality_diagnostics(clip_rows)
    summary = {
        "schema_version": "full_cv_preprocess_summary_v1",
        "run_id": run_id,
        "source_rows_total": len(source_rows),
        "source_rows_scanned": len(source_rows_to_scan),
        "download_manifest_rows": len(download_rows),
        "skipped_not_in_download_manifest": skipped_not_in_download_manifest,
        "processed_sources": processed,
        "candidate_segments": len(candidate_rows),
        "clips": len(clip_rows),
        "clean_trainable_clips": diagnostics["clean_trainable_clips"],
        "clip_quality_diagnostics": diagnostics,
        "debug_overlays": len(debug_rows),
        "skipped": skipped,
        "extract_clips": extract_clips,
        "require_non_empty": require_non_empty,
        "require_event_level_match": require_event_level_match,
        "min_video_match_confidence": min_video_match_confidence,
        "contact_search_start_sec": contact_search_start_sec,
        "contact_search_end_sec": contact_search_end_sec,
        "clip_version": clip_version,
        "resume": resume,
        "checkpoint_every_sources": checkpoint_every_sources,
        "num_workers": effective_num_workers,
        "cache_policy": cache_policy,
        "cache_stats": cache_stats,
        "progress_path": str(outputs["progress"]),
        "outputs": {key: str(value) for key, value in outputs.items() if key != "summary"},
    }
    summary["summary_path"] = str(outputs["summary"])
    outputs["summary"].parent.mkdir(parents=True, exist_ok=True)
    outputs["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if require_non_empty and not clip_rows:
        write_progress("failed_empty", len(source_rows_to_scan))
        raise RuntimeError(
            "full CV preprocessing produced 0 clips; not writing empty clip artifacts in full-run mode. "
            "Go back to notebook 11 and make sure video_sources has direct media_url rows and "
            "download_manifest has download_status=downloaded. "
            f"summary_path={outputs['summary']}"
        )

    validate_rows(CANDIDATE_SEGMENTS_SCHEMA, candidate_rows)
    validate_rows(CLIPS_SCHEMA, clip_rows)
    validate_rows(DEBUG_OVERLAYS_SCHEMA, debug_rows)
    write_table(outputs["candidate_segments"], candidate_rows)
    write_table(outputs["clips"], clip_rows)
    write_table(outputs["debug_overlays"], debug_rows)
    write_progress("complete", len(source_rows_to_scan))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight full CV preprocessing from local videos.")
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/baseball_vision")
    parser.add_argument("--run-id", default="mlb_2024_2026_full_v2")
    parser.add_argument("--bbe-events", default=None)
    parser.add_argument("--video-sources", default=None)
    parser.add_argument("--download-manifest", default=None)
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--extract-clips", action="store_true")
    parser.add_argument("--require-non-empty", action="store_true")
    parser.add_argument("--require-event-level-match", action="store_true")
    parser.add_argument("--min-video-match-confidence", type=float, default=0.45)
    parser.add_argument("--contact-search-start-sec", type=float, default=0.5)
    parser.add_argument("--contact-search-end-sec", type=float, default=8.0)
    parser.add_argument("--clip-version", default="pre_contact_long")
    parser.add_argument("--output-format", choices=("parquet", "jsonl", "json", "csv"), default="parquet")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--checkpoint-every-sources", type=int, default=10)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--cache-inputs", action="store_true")
    parser.add_argument("--cache-outputs", action="store_true")
    parser.add_argument("--cache-min-free-disk-gb", type=float, default=20.0)
    parser.add_argument("--cache-max-file-mb", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--progress-log-every-sources", type=int, default=0)
    args = parser.parse_args()
    result = run_full_cv_preprocess(
        base_dir=args.base_dir,
        run_id=args.run_id,
        bbe_events_path=args.bbe_events,
        video_sources_path=args.video_sources,
        download_manifest_path=args.download_manifest,
        max_sources=args.max_sources,
        extract_clips=args.extract_clips,
        require_non_empty=args.require_non_empty,
        require_event_level_match=args.require_event_level_match,
        min_video_match_confidence=args.min_video_match_confidence,
        contact_search_start_sec=args.contact_search_start_sec,
        contact_search_end_sec=args.contact_search_end_sec,
        clip_version=args.clip_version,
        output_suffix="." + args.output_format,
        resume=not args.no_resume,
        checkpoint_every_sources=args.checkpoint_every_sources,
        cache_dir=args.cache_dir,
        cache_inputs=args.cache_inputs,
        cache_outputs=args.cache_outputs,
        cache_min_free_disk_gb=args.cache_min_free_disk_gb,
        cache_max_file_mb=args.cache_max_file_mb,
        num_workers=args.num_workers,
        progress_log_every_sources=args.progress_log_every_sources,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
