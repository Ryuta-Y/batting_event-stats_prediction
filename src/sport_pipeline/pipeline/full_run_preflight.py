"""Full Colab run readiness checks.

This module does not download videos, train models, or mutate production
artifacts. It turns the design contract into a machine-readable checklist so a
Colab user can see which real-run stage is ready and which prerequisite is
missing.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sport_pipeline.artifact_check import check_artifacts, write_json
from sport_pipeline.colab_paths import BASE_DIR_DEFAULT, CACHE_DIR_DEFAULT, REPO_DIR_DEFAULT
from sport_pipeline.io.table import read_table
from sport_pipeline.pipeline.run_profile import DEFAULT_REAL_RUN_PROFILE, artifact_namespace, resolve_statcast_date_range
from sport_pipeline.runtime import summarize_runtime_device


DEFAULT_CONFIG = DEFAULT_REAL_RUN_PROFILE


def _load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(config_path)
    return json.loads(path.read_text(encoding="utf-8"))


def _format_artifacts(artifacts: list[str], values: dict[str, str]) -> list[str]:
    formatted = []
    for artifact in artifacts:
        try:
            formatted.append(artifact.format(**values))
        except KeyError as exc:
            missing = str(exc).strip("'")
            available = ", ".join(sorted(values))
            raise KeyError(
                f"run profile artifact placeholder {{{missing}}} is not available. "
                f"Available placeholders: {available}"
            ) from exc
    return formatted


def _status_counts(rows: list[dict[str, Any]], status_key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get(status_key) or "missing")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _read_rows_if_exists(path: Path, warnings: list[str], label: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return read_table(path)
    except Exception as exc:  # pragma: no cover - depends on optional parquet stack
        try:
            text = path.read_text(encoding="utf-8").strip()
            if text.startswith("[") or text.startswith("{"):
                payload = json.loads(text)
                if isinstance(payload, list):
                    return list(payload)
                if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
                    return list(payload["rows"])
        except Exception:
            pass
        warnings.append(f"{label} exists but could not be read: {path} ({exc})")
        return []


def build_full_run_readiness_report(
    base_dir: str | Path = BASE_DIR_DEFAULT,
    cache_dir: str | Path = CACHE_DIR_DEFAULT,
    repo_dir: str | Path = REPO_DIR_DEFAULT,
    config_path: str | Path = DEFAULT_CONFIG,
    context_run_id: str | None = None,
    full_run_id: str | None = None,
    require_gpu: bool = False,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Build and optionally persist a full-run readiness report."""

    config = _load_config(config_path)
    configured_run_ids = config.get("run_ids", {})
    resolved_context_run_id = str(
        context_run_id
        or configured_run_ids.get("recommended_context_run_id")
        or configured_run_ids.get("context_run_id")
        or "context_constant_mean_smoke"
    )
    resolved_full_run_id = str(full_run_id or configured_run_ids.get("full_run_id") or "mlb_2024_2026_full_v2")
    start_date, end_date = resolve_statcast_date_range(config)
    values = {key: str(value) for key, value in configured_run_ids.items() if value is not None}
    values.update(artifact_namespace(config))
    values.update(
        {
            "context_run_id": resolved_context_run_id,
            "full_run_id": resolved_full_run_id,
            "start_date": start_date,
            "end_date": end_date,
        }
    )
    values.setdefault(
        "sequence_run_id",
        configured_run_ids.get("sequence_run_id", "sequence_structured_mlb_2024_2026_v2"),
    )
    values.setdefault(
        "sequence_tcn_run_id",
        configured_run_ids.get("sequence_tcn_run_id", "sequence_tcn_mlb_2024_2026_v2"),
    )
    values.setdefault(
        "video_run_id",
        configured_run_ids.get(
            "video_run_id",
            configured_run_ids.get("video_frozen_run_id", "video_frozen_encoder_mlb_2024_2026_v2"),
        ),
    )
    values.setdefault(
        "video_lightweight_run_id",
        configured_run_ids.get("video_lightweight_run_id", "video_lightweight_cv2_mlb_2024_2026_v2"),
    )
    values.setdefault(
        "video_finetune_run_id",
        configured_run_ids.get("video_finetune_run_id", "video_raw_finetune_mlb_2024_2026_v2"),
    )
    values.setdefault(
        "video_ablation_report_id",
        configured_run_ids.get("video_ablation_report_id", "video_ablation_mlb_2024_2026_v2"),
    )
    values.setdefault(
        "method_evaluation_report_id",
        configured_run_ids.get("method_evaluation_report_id", "method_evaluation_mlb_2024_2026_v2"),
    )
    values.setdefault("vlm_run_id", configured_run_ids.get("vlm_run_id", "vlm_mechanics_mlb_2024_2026_v2"))
    values.setdefault("fusion_run_id", configured_run_ids.get("fusion_run_id", "fusion_mlb_2024_2026_v2"))
    values.setdefault("object_detector_run_id", configured_run_ids.get("object_detector_run_id", "bat_plate_yolo_mlb_2024_2026_v2"))
    thresholds = config.get("readiness_thresholds", {})
    strict_full_data = bool(thresholds.get("strict_full_data", False))
    artifact_groups = []
    blockers: list[str] = []
    warnings: list[str] = []
    semantic_checks: list[dict[str, Any]] = []

    for group in config["artifact_groups"]:
        artifacts = _format_artifacts(list(group["artifacts"]), values)
        check = check_artifacts(base_dir=base_dir, artifacts=artifacts)
        required = bool(group.get("required", False))
        group_status = {
            "name": group["name"],
            "required": required,
            "stage": group.get("stage"),
            "description_ja": group.get("description_ja"),
            "all_present": check["all_present"],
            "artifacts": check["artifacts"],
        }
        artifact_groups.append(group_status)
        if not check["all_present"]:
            missing = [entry["artifact"] for entry in check["artifacts"] if not entry["exists"]]
            message = f"{group['name']} missing: {', '.join(missing)}"
            if required:
                blockers.append(message)
            else:
                warnings.append(message)

    device = summarize_runtime_device(prefer_gpu=True, require_gpu=require_gpu)
    if require_gpu and device["selected_device"] != "cuda":
        blockers.append("GPU required but CUDA is not available")
    if device.get("warning_ja"):
        warnings.append(str(device["warning_ja"]))

    repo = Path(repo_dir)
    base = Path(base_dir)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path_warnings = []
    if not repo.exists():
        path_warnings.append(f"REPO_DIR not found: {repo}")
    if not base.exists():
        path_warnings.append(f"BASE_DIR not found: {base}")
    if not cache.exists():
        path_warnings.append(f"CACHE_DIR not found: {cache}")
    warnings.extend(path_warnings)

    bbe_events_path = base / "manifests/bbe_events_v1.parquet"
    bbe_rows = _read_rows_if_exists(bbe_events_path, warnings, "bbe_events_v1")
    if bbe_rows:
        n_home_runs = sum(1 for row in bbe_rows if bool(row.get("is_home_run")))
        dataset_role_counts = _status_counts(bbe_rows, "dataset_role")
        seasons = sorted({int(row["season"]) for row in bbe_rows if row.get("season") is not None})
        bbe_check = {
            "name": "bbe_events_semantics",
            "path": str(bbe_events_path),
            "n_rows": len(bbe_rows),
            "seasons": seasons,
            "n_home_runs": n_home_runs,
            "dataset_role_counts": dataset_role_counts,
            "n_target_xba_available": sum(1 for row in bbe_rows if bool(row.get("target_xba_available"))),
            "n_target_xwoba_available": sum(1 for row in bbe_rows if bool(row.get("target_xwoba_available"))),
            "n_target_ops_available": sum(1 for row in bbe_rows if bool(row.get("target_ops_available"))),
        }
        semantic_checks.append(bbe_check)
        min_bbe_rows = int(thresholds.get("min_bbe_rows", 100))
        if len(bbe_rows) < min_bbe_rows:
            message = (
                f"bbe_events_v1 has only {len(bbe_rows)} rows; expected at least "
                f"{min_bbe_rows} rows for this real-run profile"
            )
            (blockers if strict_full_data else warnings).append(message)
        expected_seasons = set(config.get("data_window", {}).get("expected_seasons", []))
        if expected_seasons and not expected_seasons.issubset(set(seasons)):
            warnings.append(f"bbe_events_v1 seasons {seasons} do not cover expected seasons {sorted(expected_seasons)}")
        if n_home_runs and n_home_runs == len(bbe_rows):
            blockers.append("bbe_events_v1 appears home-run-only; rebuild from Statcast BBE/PA universe first")

    video_sources_path = base / "manifests/video_sources_v1.parquet"
    video_source_rows = _read_rows_if_exists(video_sources_path, warnings, "video_sources_v1")
    if video_source_rows:
        media_url_count = sum(1 for row in video_source_rows if row.get("media_url"))
        source_topic_counts = _status_counts(video_source_rows, "source_topic")
        download_status_counts = _status_counts(video_source_rows, "download_status")
        video_source_check = {
            "name": "video_sources_semantics",
            "path": str(video_sources_path),
            "n_rows": len(video_source_rows),
            "n_media_url": media_url_count,
            "source_topic_counts": source_topic_counts,
            "download_status_counts": download_status_counts,
        }
        semantic_checks.append(video_source_check)
        min_video_sources = int(thresholds.get("min_video_sources", 0))
        min_media_url_rows = int(thresholds.get("min_media_url_rows", 0))
        if min_video_sources and len(video_source_rows) < min_video_sources:
            message = f"video_sources_v1 has only {len(video_source_rows)} rows; expected at least {min_video_sources}"
            (blockers if strict_full_data else warnings).append(message)
        if media_url_count < min_media_url_rows:
            message = (
                f"video_sources_v1 has {media_url_count} direct media_url rows; "
                f"expected at least {min_media_url_rows} before full CV download"
            )
            (blockers if strict_full_data else warnings).append(message)
        home_run_like = sum(1 for row in video_source_rows if "home" in str(row.get("source_topic", "")).lower())
        if home_run_like and home_run_like == len(video_source_rows):
            blockers.append("video_sources_v1 appears home-run-only; do not use as the event universe")

    download_manifest_path = base / f"raw_videos/{resolved_full_run_id}/download_manifest_v1.parquet"
    download_rows = _read_rows_if_exists(download_manifest_path, warnings, "download_manifest_v1")
    if download_rows:
        download_status_counts = _status_counts(download_rows, "download_status")
        skip_reason_counts = _status_counts(download_rows, "skip_reason")
        downloaded_count = int(download_status_counts.get("downloaded", 0))
        planned_count = int(download_status_counts.get("planned", 0))
        download_check = {
            "name": "download_manifest_semantics",
            "path": str(download_manifest_path),
            "n_rows": len(download_rows),
            "download_status_counts": download_status_counts,
            "skip_reason_counts": skip_reason_counts,
            "n_downloaded": downloaded_count,
            "n_planned": planned_count,
        }
        semantic_checks.append(download_check)
        if downloaded_count == 0:
            warnings.append(
                "download_manifest_v1 exists but has zero downloaded videos; clips_v1 cannot be built from local media yet"
            )
        min_downloaded_videos = int(thresholds.get("min_downloaded_videos", 0))
        if downloaded_count and downloaded_count < min_downloaded_videos:
            warnings.append(
                f"download_manifest_v1 has only {downloaded_count} downloaded videos; "
                f"real profile target is at least {min_downloaded_videos}"
            )
        if skip_reason_counts.get("missing_media_url", 0) == len(download_rows):
            warnings.append(
                "all download_manifest_v1 rows skipped with missing_media_url; add direct media URLs or a source-specific resolver"
            )

    payload = {
        "schema_version": "full_run_readiness_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "full_run_id": resolved_full_run_id,
        "context_run_id": resolved_context_run_id,
        "config_path": str(config_path),
        "statcast_date_range": {"start_date": start_date, "end_date": end_date},
        "paths": {
            "repo_dir": str(repo),
            "base_dir": str(base),
            "cache_dir": str(cache),
        },
        "device": device,
        "artifact_groups": artifact_groups,
        "semantic_checks": semantic_checks,
        "heavy_steps": config["heavy_steps"],
        "recommended_gpu_ja": config["recommended_gpu_ja"],
        "blockers": blockers,
        "warnings": warnings,
        "ready_for_full_video_sequence_fusion": len(blockers) == 0,
    }
    if output_json is not None:
        write_json(payload, output_json)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Check full Colab run readiness.")
    parser.add_argument("--base-dir", default=str(BASE_DIR_DEFAULT))
    parser.add_argument("--cache-dir", default=str(CACHE_DIR_DEFAULT))
    parser.add_argument("--repo-dir", default=str(REPO_DIR_DEFAULT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--context-run-id", default=None)
    parser.add_argument("--full-run-id", default=None)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    result = build_full_run_readiness_report(
        base_dir=args.base_dir,
        cache_dir=args.cache_dir,
        repo_dir=args.repo_dir,
        config_path=args.config,
        context_run_id=args.context_run_id,
        full_run_id=args.full_run_id,
        require_gpu=args.require_gpu,
        output_json=args.output_json,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
