"""Build video_sources_v1 candidate rows from a BBE manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sport_pipeline.io import read_table, write_table
from sport_pipeline.schemas import VIDEO_SOURCES_SCHEMA
from sport_pipeline.schemas.data_manifest import validate_rows


def _candidate_row(event: dict[str, Any], candidate_rank: int = 1) -> dict[str, Any]:
    event_id = str(event["event_id"])
    source_url = event.get("source_url") or event.get("video_url")
    media_url = event.get("media_url")
    video_available = bool(source_url or media_url or event.get("has_video_candidate"))
    source_topic = str(event.get("source_topic") or ("home_runs" if event.get("is_home_run") and video_available else "statcast_bbe_search"))
    dataset_role = "smoke_test" if source_topic == "home_runs" else str(event.get("dataset_role", "train_candidate"))
    return {
        "schema_version": VIDEO_SOURCES_SCHEMA.version,
        "video_source_id": f"vs_{event_id}_{candidate_rank}",
        "event_id": event_id,
        "same_event_group_id": str(event.get("same_event_group_id", event_id)),
        "source_video_id": event.get("source_video_id"),
        "view_id": f"{event_id}_view{candidate_rank}",
        "source_kind": str(event.get("source_kind") or ("manual_reference" if source_url else "other")),
        "source_url": source_url,
        "media_url": media_url,
        "source_topic": source_topic,
        "dataset_role": dataset_role,
        "rights_status": str(event.get("rights_status") or ("official_public_reference" if source_url else "check_required")),
        "match_confidence": float(event.get("match_confidence") or (0.5 if video_available else 0.0)),
        "match_reason": str(event.get("match_reason") or ("manifest_video_reference" if video_available else "no_candidate_yet")),
        "join_key_fields": event.get("join_key_fields") or ["game_pk", "at_bat_number", "pitch_number"],
        "candidate_rank": candidate_rank,
        "video_available": video_available,
        "download_status": str(event.get("download_status") or ("not_attempted" if video_available else "blocked")),
        "local_video_path": event.get("local_video_path"),
        "probe_status": str(event.get("probe_status") or ("pending" if video_available else "review_only")),
        "review_status": str(event.get("review_status") or ("pending" if video_available else "review_only")),
        "reject_reason": event.get("reject_reason"),
        "view_label": str(event.get("view_label") or "unknown"),
        "view_confidence": float(event.get("view_confidence") or (0.5 if video_available else 0.0)),
        "batting_visibility": str(event.get("batting_visibility") or "unknown"),
        "is_replay": bool(event.get("is_replay", False)),
        "is_non_batting_segment": bool(event.get("is_non_batting_segment", False)),
    }


def build_video_source_rows(bbe_rows: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    """Create candidate source rows without downloading media."""

    rows = [_candidate_row(event) for event in bbe_rows]
    if limit is not None:
        rows = rows[:limit]
    validate_rows(VIDEO_SOURCES_SCHEMA, rows)
    return rows


def build_video_source_artifact(
    base_dir: str | Path,
    bbe_path: str | Path,
    *,
    limit: int | None = None,
    output_suffix: str = ".parquet",
) -> Path:
    """Write video_sources_v1 under BASE_DIR."""

    bbe_rows = read_table(bbe_path)
    rows = build_video_source_rows(bbe_rows, limit=limit)
    output = Path(base_dir) / f"manifests/video_sources_v1{output_suffix}"
    write_table(output, rows)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build video_sources_v1 candidate manifest without downloads.")
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/baseball_vision")
    parser.add_argument("--bbe-events", default=None, help="Input bbe_events_v1 table. Defaults to BASE_DIR/manifests/bbe_events_v1.parquet")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-format", choices=("parquet", "jsonl", "json", "csv"), default="parquet")
    args = parser.parse_args(argv)
    suffix = "." + args.output_format
    bbe_path = Path(args.bbe_events) if args.bbe_events else Path(args.base_dir) / "manifests/bbe_events_v1.parquet"
    output = build_video_source_artifact(args.base_dir, bbe_path, limit=args.limit, output_suffix=suffix)
    print(json.dumps({"video_sources": str(output)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
