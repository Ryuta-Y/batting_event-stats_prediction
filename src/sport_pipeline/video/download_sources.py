"""Direct media-url downloader for Colab-side video source artifacts.

This module intentionally does not scrape pages or bypass access controls. It
only handles rows that already expose a direct `media_url` in
`video_sources_v1`, writes under the Drive artifact root, and defaults to a
dry-run plan unless `execute=True` or `--execute` is provided.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from sport_pipeline.io.table import read_table, write_table
from sport_pipeline.video.source_quality import source_priority, video_source_skip_reason


DEFAULT_ALLOWED_RIGHTS = (
    "public_domain",
    "open_dataset",
    "official_public_reference",
    "personal_research_only",
)


@dataclass(frozen=True)
class DownloadPlanRow:
    """One planned, skipped, downloaded, or failed source download row."""

    video_source_id: str
    event_id: str | None
    source_url: str | None
    media_url: str | None
    rights_status: str
    planned_path: str | None
    download_status: str
    skip_reason: str | None = None
    size_bytes: int | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        """Return JSON-serializable metadata."""

        return asdict(self)


def _allowed(rights_status: str, allowed_rights: Iterable[str]) -> bool:
    return rights_status in set(allowed_rights)


def _extension_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix:
        return suffix
    guessed = mimetypes.guess_extension("video/mp4")
    return guessed or ".mp4"


def _safe_source_id(row: dict, index: int) -> str:
    raw = str(row.get("video_source_id") or row.get("source_video_id") or f"source_{index:05d}")
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in raw)


def plan_video_downloads(
    video_sources_path: str | Path,
    output_dir: str | Path,
    allowed_rights: Iterable[str] = DEFAULT_ALLOWED_RIGHTS,
    max_files: int | None = None,
    batch_count: int | None = None,
    batch_index: int = 0,
    include_skipped: bool = True,
    require_event_level_match: bool = False,
    min_match_confidence: float = 0.45,
    prefer_exact_play: bool = False,
    max_sources_per_event: int | None = None,
) -> list[DownloadPlanRow]:
    """Build a dry-run download plan from `video_sources_v1` rows."""

    if batch_count is not None and batch_count < 1:
        raise ValueError("batch_count must be >= 1")
    if batch_count is not None and not 0 <= batch_index < batch_count:
        raise ValueError("batch_index must satisfy 0 <= batch_index < batch_count")
    rows = read_table(video_sources_path)
    if prefer_exact_play:
        rows = sorted(rows, key=source_priority)
    output_root = Path(output_dir)
    planned: list[DownloadPlanRow] = []
    eligible_rank = 0
    planned_by_event: dict[str, int] = {}
    for index, row in enumerate(rows):
        source_id = _safe_source_id(row, index)
        media_url = row.get("media_url")
        source_url = row.get("source_url")
        rights_status = str(row.get("rights_status") or "unknown")
        event_id = row.get("event_id")
        event_key = str(event_id) if event_id is not None else ""
        if not media_url:
            if include_skipped:
                planned.append(
                    DownloadPlanRow(
                        video_source_id=source_id,
                        event_id=str(event_id) if event_id is not None else None,
                        source_url=str(source_url) if source_url else None,
                        media_url=None,
                        rights_status=rights_status,
                        planned_path=None,
                        download_status="skipped",
                        skip_reason="missing_media_url",
                    )
                )
            continue
        if require_event_level_match:
            skip_reason = video_source_skip_reason(row, min_match_confidence=min_match_confidence)
            if skip_reason is not None:
                if include_skipped:
                    planned.append(
                        DownloadPlanRow(
                            video_source_id=source_id,
                            event_id=str(event_id) if event_id is not None else None,
                            source_url=str(source_url) if source_url else None,
                            media_url=str(media_url),
                            rights_status=rights_status,
                            planned_path=None,
                            download_status="skipped",
                            skip_reason=skip_reason,
                        )
                    )
                continue
        if not _allowed(rights_status, allowed_rights):
            if include_skipped:
                planned.append(
                    DownloadPlanRow(
                        video_source_id=source_id,
                        event_id=str(event_id) if event_id is not None else None,
                        source_url=str(source_url) if source_url else None,
                        media_url=str(media_url),
                        rights_status=rights_status,
                        planned_path=None,
                        download_status="skipped",
                        skip_reason=f"rights_status_not_allowed:{rights_status}",
                    )
                )
            continue
        if max_sources_per_event is not None and event_key:
            event_count = planned_by_event.get(event_key, 0)
            if event_count >= max_sources_per_event:
                if include_skipped:
                    planned.append(
                        DownloadPlanRow(
                            video_source_id=source_id,
                            event_id=str(event_id) if event_id is not None else None,
                            source_url=str(source_url) if source_url else None,
                            media_url=str(media_url),
                            rights_status=rights_status,
                            planned_path=None,
                            download_status="skipped",
                            skip_reason=f"max_sources_per_event_reached:{max_sources_per_event}",
                        )
                    )
                continue
        current_rank = eligible_rank
        eligible_rank += 1
        if max_files is not None and current_rank >= max_files:
            if include_skipped:
                planned.append(
                    DownloadPlanRow(
                        video_source_id=source_id,
                        event_id=str(event_id) if event_id is not None else None,
                        source_url=str(source_url) if source_url else None,
                        media_url=str(media_url),
                        rights_status=rights_status,
                        planned_path=None,
                        download_status="skipped",
                        skip_reason="max_files_reached",
                    )
                )
            continue
        if batch_count is not None and current_rank % batch_count != batch_index:
            if include_skipped:
                planned.append(
                    DownloadPlanRow(
                        video_source_id=source_id,
                        event_id=str(event_id) if event_id is not None else None,
                        source_url=str(source_url) if source_url else None,
                        media_url=str(media_url),
                        rights_status=rights_status,
                        planned_path=None,
                        download_status="skipped",
                        skip_reason=f"outside_batch:{batch_index + 1}_of_{batch_count}",
                    )
                )
            continue
        path = output_root / f"{source_id}{_extension_from_url(str(media_url))}"
        if event_key:
            planned_by_event[event_key] = planned_by_event.get(event_key, 0) + 1
        planned.append(
            DownloadPlanRow(
                video_source_id=source_id,
                event_id=str(event_id) if event_id is not None else None,
                source_url=str(source_url) if source_url else None,
                media_url=str(media_url),
                rights_status=rights_status,
                planned_path=str(path),
                download_status="planned",
            )
        )
    return planned


def _manifest_rows_to_write(rows_by_id: dict[str, dict]) -> list[dict]:
    """Keep the download manifest compact enough for repeated Drive writes."""

    rows = []
    for row in rows_by_id.values():
        status = row.get("download_status")
        if status in {"planned", "downloaded", "failed"} or row.get("planned_path"):
            rows.append(row)
    return sorted(rows, key=lambda item: str(item.get("video_source_id") or ""))


def _download_progress_bar(current: int, total: int, width: int = 28) -> str:
    """Return a compact ASCII progress bar for notebook logs."""

    if total <= 0:
        return "[----------------------------] 0.0%"
    clamped = max(0, min(current, total))
    filled = round(width * clamped / total)
    percent = 100.0 * clamped / total
    return f"[{'#' * filled}{'-' * (width - filled)}] {percent:5.1f}%"


def summarize_download_manifest(path: str | Path) -> dict[str, object]:
    """Return a small progress summary for an existing download manifest."""

    manifest = Path(path)
    if not manifest.exists():
        return {"exists": False, "rows": 0, "status_counts": {}, "downloaded": 0, "failed": 0}
    rows = read_table(manifest)
    status_counts: dict[str, int] = {}
    total_bytes = 0
    for row in rows:
        status = str(row.get("download_status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        try:
            total_bytes += int(row.get("size_bytes") or 0)
        except (TypeError, ValueError):
            pass
    return {
        "exists": True,
        "rows": len(rows),
        "status_counts": status_counts,
        "downloaded": status_counts.get("downloaded", 0),
        "failed": status_counts.get("failed", 0),
        "total_size_bytes": total_bytes,
        "path": str(manifest),
    }


def _copy_url_to_path(url: str, path: Path, timeout_sec: int, max_bytes: int | None) -> int:
    request = urllib.request.Request(url, headers={"User-Agent": "sport-pipeline-research/1.0"})
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with urllib.request.urlopen(request, timeout=timeout_sec) as response, path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if max_bytes is not None and written > max_bytes:
                raise RuntimeError(f"download exceeds max_bytes_per_file={max_bytes}")
            handle.write(chunk)
    return written


def _download_hls_with_ffmpeg(url: str, path: Path, timeout_sec: int) -> int:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to download HLS .m3u8 media")
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        url,
        "-c",
        "copy",
        str(path),
    ]
    subprocess.run(cmd, check=True, timeout=timeout_sec)
    return path.stat().st_size


def _execute_download_plan_row(
    row: DownloadPlanRow,
    execute: bool,
    max_bytes_per_file: int | None,
    timeout_sec: int,
    allow_hls: bool,
) -> dict:
    """Execute one planned row and return its manifest payload."""

    payload = row.to_dict()
    if not execute or row.download_status != "planned" or row.planned_path is None or row.media_url is None:
        return payload

    path = Path(row.planned_path)
    effective_path = path
    try:
        effective_path = path.with_suffix(".mp4") if row.media_url and ".m3u8" in row.media_url.lower() else path
        if effective_path.exists():
            payload["download_status"] = "downloaded"
            payload["planned_path"] = str(effective_path)
            payload["size_bytes"] = effective_path.stat().st_size
            payload["skip_reason"] = "already_exists"
        else:
            if ".m3u8" in row.media_url.lower():
                if not allow_hls:
                    raise RuntimeError("HLS .m3u8 download disabled")
                payload["size_bytes"] = _download_hls_with_ffmpeg(row.media_url, effective_path, timeout_sec)
                payload["planned_path"] = str(effective_path)
            else:
                payload["size_bytes"] = _copy_url_to_path(
                    row.media_url,
                    effective_path,
                    timeout_sec=timeout_sec,
                    max_bytes=max_bytes_per_file,
                )
            payload["download_status"] = "downloaded"
    except (urllib.error.URLError, OSError, RuntimeError, ValueError) as exc:
        if effective_path.exists():
            try:
                effective_path.unlink()
            except OSError:
                pass
        payload["download_status"] = "failed"
        payload["error"] = str(exc)
    return payload


_DOWNLOAD_PATH_KEYS = ("planned_path", "local_video_path", "video_path", "path")


def _existing_download_path(row: dict | None) -> Path | None:
    """Return the first existing local video path recorded in a manifest row."""

    if not isinstance(row, dict):
        return None
    for key in _DOWNLOAD_PATH_KEYS:
        value = row.get(key)
        if not value:
            continue
        path = Path(str(value))
        if path.exists():
            return path
    return None


def _is_reusable_download_row(row: dict | None) -> bool:
    return bool(
        isinstance(row, dict)
        and str(row.get("download_status") or "") == "downloaded"
        and _existing_download_path(row) is not None
    )


def _normalise_reuse_row(
    row: dict,
    *,
    source_manifest: str | None = None,
    source_run_id: str | None = None,
    skip_reason: str = "reused_existing_file",
) -> dict:
    payload = dict(row)
    existing_path = _existing_download_path(payload)
    if existing_path is not None:
        payload["planned_path"] = str(existing_path)
        payload["size_bytes"] = payload.get("size_bytes") or existing_path.stat().st_size
    payload["download_status"] = "downloaded"
    payload["skip_reason"] = payload.get("skip_reason") or skip_reason
    if source_manifest:
        payload["reuse_source_manifest"] = source_manifest
    if source_run_id:
        payload["reuse_source_run_id"] = source_run_id
    return payload


def _load_reusable_manifest_rows(
    manifest_paths: Iterable[str | Path] | None,
) -> tuple[dict[str, dict], list[str], list[str], int]:
    rows_by_id: dict[str, dict] = {}
    source_paths: list[str] = []
    missing_paths: list[str] = []
    ignored_rows = 0
    for manifest in manifest_paths or []:
        path = Path(manifest)
        source_paths.append(str(path))
        if not path.exists():
            missing_paths.append(str(path))
            continue
        for row in read_table(path):
            source_id = str(row.get("video_source_id") or "")
            if not source_id or not _is_reusable_download_row(row):
                ignored_rows += 1
                continue
            rows_by_id[source_id] = _normalise_reuse_row(row, source_manifest=str(path))
    return rows_by_id, source_paths, missing_paths, ignored_rows


def _download_manifest_candidates(run_dir: Path, manifest_name: str) -> list[Path]:
    names = [
        manifest_name,
        "download_manifest_v1.parquet",
        "download_manifest_v1.jsonl",
        "download_manifest_v1.json",
        "download_manifest_v1.csv",
    ]
    seen: set[str] = set()
    candidates: list[Path] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        candidates.append(run_dir / name)
    return candidates


def seed_download_manifest_from_previous_runs(
    base_dir: str | Path,
    target_run_id: str,
    source_run_ids: Iterable[str],
    *,
    output_manifest: str | Path | None = None,
    manifest_name: str = "download_manifest_v1.parquet",
    summary_path: str | Path | None = None,
) -> dict[str, object]:
    """Seed a target run manifest with reusable downloaded videos from prior runs.

    The function does not copy media bytes. It writes manifest rows whose
    ``planned_path`` still points at the existing previous-run video file, so a
    v2 run can reuse v1 media while writing all derived clips/features/reports
    into v2-scoped artifact folders.
    """

    base = Path(base_dir)
    source_run_id_list = [str(item) for item in source_run_ids]
    output = Path(output_manifest) if output_manifest is not None else base / "raw_videos" / target_run_id / manifest_name
    rows_by_id: dict[str, dict] = {}
    source_manifests: list[str] = []
    missing_manifests: list[str] = []
    ignored_rows = 0
    for source_run in source_run_id_list:
        run_dir = base / "raw_videos" / source_run
        candidates = _download_manifest_candidates(run_dir, manifest_name)
        source_manifest = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        source_manifests.append(str(source_manifest))
        if not source_manifest.exists():
            missing_manifests.append(str(source_manifest))
            continue
        for row in read_table(source_manifest):
            source_id = str(row.get("video_source_id") or "")
            if not source_id or not _is_reusable_download_row(row):
                ignored_rows += 1
                continue
            rows_by_id[source_id] = _normalise_reuse_row(
                row,
                source_manifest=str(source_manifest),
                source_run_id=source_run,
            )

    write_table(output, _manifest_rows_to_write(rows_by_id))
    summary = summarize_download_manifest(output)
    summary.update(
        {
            "schema_version": "download_manifest_reuse_seed_v1",
            "target_run_id": target_run_id,
            "source_run_ids": source_run_id_list,
            "source_manifests": source_manifests,
            "missing_manifests": missing_manifests,
            "ignored_rows": ignored_rows,
            "copied_media_files": 0,
            "note": "Media files are reused by path; this step writes only the target manifest.",
        }
    )
    if summary_path is not None:
        summary_output = Path(summary_path)
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def merge_download_manifests(manifest_paths: Iterable[str | Path], output_manifest: str | Path) -> dict[str, object]:
    """Merge shard download manifests into the canonical manifest."""

    rows_by_id: dict[str, dict] = {}
    source_paths: list[str] = []
    missing_paths: list[str] = []
    for manifest in manifest_paths:
        path = Path(manifest)
        source_paths.append(str(path))
        if not path.exists():
            missing_paths.append(str(path))
            continue
        for row in read_table(path):
            source_id = str(row.get("video_source_id") or "")
            if source_id:
                rows_by_id[source_id] = row
    output = Path(output_manifest)
    rows_to_write = _manifest_rows_to_write(rows_by_id)
    write_table(output, rows_to_write)
    summary = summarize_download_manifest(output)
    summary["source_manifests"] = source_paths
    summary["missing_manifests"] = missing_paths
    return summary


def download_video_sources(
    video_sources_path: str | Path,
    output_dir: str | Path,
    output_manifest: str | Path | None = None,
    reuse_manifest_paths: Iterable[str | Path] | None = None,
    execute: bool = False,
    allowed_rights: Iterable[str] = DEFAULT_ALLOWED_RIGHTS,
    max_files: int | None = None,
    max_bytes_per_file: int | None = None,
    timeout_sec: int = 60,
    allow_hls: bool = True,
    batch_count: int | None = None,
    batch_index: int = 0,
    progress_path: str | Path | None = None,
    log_every: int = 1,
    manifest_write_every: int = 1,
    num_workers: int = 1,
    require_event_level_match: bool = False,
    min_match_confidence: float = 0.45,
    prefer_exact_play: bool = False,
    max_sources_per_event: int | None = None,
) -> list[dict]:
    """Plan or execute direct media-url downloads.

    When `execute=False`, the function writes only a plan and performs no
    network IO. This is the default so notebooks can show exactly what would
    happen before a user spends Colab time or downloads large files.
    """

    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")

    plan = plan_video_downloads(
        video_sources_path=video_sources_path,
        output_dir=output_dir,
        allowed_rights=allowed_rights,
        max_files=max_files,
        batch_count=batch_count,
        batch_index=batch_index,
        include_skipped=False,
        require_event_level_match=require_event_level_match,
        min_match_confidence=min_match_confidence,
        prefer_exact_play=prefer_exact_play,
        max_sources_per_event=max_sources_per_event,
    )
    overall_plan = plan_video_downloads(
        video_sources_path=video_sources_path,
        output_dir=output_dir,
        allowed_rights=allowed_rights,
        max_files=max_files,
        batch_count=None,
        include_skipped=False,
        require_event_level_match=require_event_level_match,
        min_match_confidence=min_match_confidence,
        prefer_exact_play=prefer_exact_play,
        max_sources_per_event=max_sources_per_event,
    )
    overall_source_ids = {row.video_source_id for row in overall_plan}
    overall_total = len(overall_source_ids)
    manifest_path = Path(output_manifest) if output_manifest is not None else None
    progress = Path(progress_path) if progress_path is not None else None
    results_by_id: dict[str, dict] = {}
    if manifest_path is not None and manifest_path.exists():
        for row in read_table(manifest_path):
            source_id = str(row.get("video_source_id") or "")
            if source_id:
                results_by_id[source_id] = (
                    _normalise_reuse_row(row, skip_reason="already_exists")
                    if _is_reusable_download_row(row)
                    else row
                )
    reuse_rows, reuse_source_paths, reuse_missing_paths, reuse_ignored_rows = _load_reusable_manifest_rows(reuse_manifest_paths)
    reuse_seeded_count = 0
    for source_id, row in reuse_rows.items():
        current = results_by_id.get(source_id)
        if _is_reusable_download_row(current):
            continue
        results_by_id[source_id] = row
        reuse_seeded_count += 1
    reused_existing_this_call = sum(1 for row in plan if _is_reusable_download_row(results_by_id.get(row.video_source_id)))
    pending_plan = [row for row in plan if not _is_reusable_download_row(results_by_id.get(row.video_source_id))]

    started_at = time.time()

    def compact_results() -> list[dict]:
        return _manifest_rows_to_write(results_by_id)

    def overall_status_counts() -> dict[str, int]:
        status_counts: dict[str, int] = {}
        for source_id in overall_source_ids:
            row = results_by_id.get(source_id)
            if row is None:
                continue
            status = str(row.get("download_status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        return status_counts

    def overall_completed_count() -> int:
        counts = overall_status_counts()
        return counts.get("downloaded", 0) + counts.get("failed", 0)

    def write_outputs(processed: int, last_row: dict | None = None) -> None:
        processed_or_reused = processed + reused_existing_this_call
        rows_to_write = compact_results()
        if manifest_path is not None:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            write_table(manifest_path, rows_to_write)
        if progress is not None:
            status_counts: dict[str, int] = {}
            for item in rows_to_write:
                status = str(item.get("download_status") or "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
            overall_counts = overall_status_counts()
            overall_completed = overall_counts.get("downloaded", 0) + overall_counts.get("failed", 0)
            overall_percent = 100.0 * overall_completed / overall_total if overall_total else 0.0
            payload = {
                "schema_version": "video_download_progress_v1",
                "status": "running" if processed_or_reused < len(plan) else "complete",
                "overall_status": "complete" if overall_total and overall_completed >= overall_total else "incomplete",
                "execute": execute,
                "video_sources_path": str(video_sources_path),
                "output_dir": str(output_dir),
                "output_manifest": str(manifest_path) if manifest_path is not None else None,
                "reuse_manifest_paths": reuse_source_paths,
                "reuse_missing_manifest_paths": reuse_missing_paths,
                "reuse_seeded_count": reuse_seeded_count,
                "reuse_ignored_rows": reuse_ignored_rows,
                "reused_existing_this_call": reused_existing_this_call,
                "batch_index": batch_index,
                "batch_count": batch_count,
                "max_files": max_files,
                "num_workers": num_workers,
                "require_event_level_match": require_event_level_match,
                "min_match_confidence": min_match_confidence,
                "prefer_exact_play": prefer_exact_play,
                "max_sources_per_event": max_sources_per_event,
                "total_planned_this_call": len(plan),
                "processed_this_call": processed_or_reused,
                "downloaded_or_failed_this_runtime": processed,
                "remaining_this_call": max(0, len(plan) - processed_or_reused),
                "overall_total_planned": overall_total,
                "overall_completed": overall_completed,
                "overall_remaining": max(0, overall_total - overall_completed),
                "overall_percent": round(overall_percent, 3),
                "overall_progress_bar": _download_progress_bar(overall_completed, overall_total),
                "overall_status_counts": overall_counts,
                "manifest_rows": len(rows_to_write),
                "status_counts": status_counts,
                "elapsed_sec": round(time.time() - started_at, 3),
                "last_video_source_id": last_row.get("video_source_id") if last_row else None,
                "last_status": last_row.get("download_status") if last_row else None,
                "last_error": last_row.get("error") if last_row else None,
            }
            progress.parent.mkdir(parents=True, exist_ok=True)
            progress.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def record_result(processed: int, payload: dict) -> None:
        results_by_id[payload["video_source_id"]] = payload
        if log_every > 0 and (processed == 1 or processed % log_every == 0 or payload["download_status"] == "failed"):
            overall_completed = overall_completed_count()
            overall_digits = max(1, len(str(overall_total)))
            processed_or_reused = processed + reused_existing_this_call
            print(
                "video_download",
                f"{overall_completed:>{overall_digits}}/{overall_total}",
                _download_progress_bar(overall_completed, overall_total),
                f"batch={batch_index + 1 if batch_count else 1}/{batch_count or 1}",
                f"batch_item={processed_or_reused}/{len(plan)}",
                payload["video_source_id"],
                payload["download_status"],
                payload.get("size_bytes"),
                payload.get("skip_reason") or payload.get("error") or "",
                flush=True,
            )
        if processed % max(1, manifest_write_every) == 0:
            write_outputs(processed, payload)

    if execute and num_workers > 1 and len(pending_plan) > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    _execute_download_plan_row,
                    row,
                    execute,
                    max_bytes_per_file,
                    timeout_sec,
                    allow_hls,
                ): row
                for row in pending_plan
            }
            for processed, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                try:
                    payload = future.result()
                except Exception as exc:  # pragma: no cover - defensive guard around worker futures.
                    payload = row.to_dict()
                    payload["download_status"] = "failed"
                    payload["error"] = str(exc)
                record_result(processed, payload)
    else:
        for processed, row in enumerate(pending_plan, start=1):
            payload = _execute_download_plan_row(
                row,
                execute,
                max_bytes_per_file=max_bytes_per_file,
                timeout_sec=timeout_sec,
                allow_hls=allow_hls,
            )
            record_result(processed, payload)
    write_outputs(len(pending_plan), None)
    return compact_results()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or execute direct media-url video downloads.")
    parser.add_argument("--video-sources", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-manifest", default=None)
    parser.add_argument(
        "--reuse-manifest",
        action="append",
        default=[],
        help="Existing download manifest to reuse before downloading missing rows. Can be passed multiple times.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually download planned media URLs.")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-bytes-per-file", type=int, default=None)
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--no-hls", action="store_true", help="Disable ffmpeg-based .m3u8 downloads.")
    parser.add_argument("--batch-count", type=int, default=None)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--progress-path", default=None)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--manifest-write-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--require-event-level-match", action="store_true")
    parser.add_argument("--min-match-confidence", type=float, default=0.45)
    parser.add_argument("--prefer-exact-play", action="store_true")
    parser.add_argument("--max-sources-per-event", type=int, default=None)
    parser.add_argument(
        "--allowed-rights",
        default=",".join(DEFAULT_ALLOWED_RIGHTS),
        help="Comma-separated rights_status allowlist.",
    )
    args = parser.parse_args()
    result = download_video_sources(
        video_sources_path=args.video_sources,
        output_dir=args.output_dir,
        output_manifest=args.output_manifest,
        reuse_manifest_paths=args.reuse_manifest,
        execute=args.execute,
        allowed_rights=tuple(part.strip() for part in args.allowed_rights.split(",") if part.strip()),
        max_files=args.max_files,
        max_bytes_per_file=args.max_bytes_per_file,
        timeout_sec=args.timeout_sec,
        allow_hls=not args.no_hls,
        batch_count=args.batch_count,
        batch_index=args.batch_index,
        progress_path=args.progress_path,
        log_every=args.log_every,
        manifest_write_every=args.manifest_write_every,
        num_workers=args.num_workers,
        require_event_level_match=args.require_event_level_match,
        min_match_confidence=args.min_match_confidence,
        prefer_exact_play=args.prefer_exact_play,
        max_sources_per_event=args.max_sources_per_event,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
