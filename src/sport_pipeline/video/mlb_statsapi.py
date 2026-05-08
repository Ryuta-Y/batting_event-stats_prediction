"""MLB Stats API game-content video source resolver.

This resolver searches per-game content JSON for video/media assets and maps
them back to BBE events by conservative text evidence. It does not download
media and it does not bypass access controls. The output is still
`video_sources_v1`: candidate evidence, not proof of a clean clip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from sport_pipeline.io import read_table, write_table
from sport_pipeline.probe_video_sources import build_video_source_rows
from sport_pipeline.schemas import VIDEO_SOURCES_SCHEMA
from sport_pipeline.schemas.data_manifest import validate_rows
from sport_pipeline.video.source_quality import has_event_level_match_reason


STATSAPI_GAME_CONTENT_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/content"
BASEBALL_SAVANT_SPORTY_URL = "https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"


@dataclass(frozen=True)
class VideoAsset:
    """One candidate media asset from game content JSON."""

    source_video_id: str
    title: str
    description: str
    source_url: str | None
    media_url: str | None
    duration_sec: float | None
    published_date: str | None
    raw_path: str


def _fetch_json(url: str, timeout_sec: int = 60) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "sport-pipeline-research/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_text(url: str, timeout_sec: int = 60) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "sport-pipeline-research/1.0",
            "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        return response.read().decode("utf-8", errors="replace")


def fetch_game_content(game_pk: int, cache_dir: str | Path | None = None, timeout_sec: int = 60) -> dict[str, Any]:
    """Fetch or read cached MLB Stats API game content JSON."""

    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"game_content_{game_pk}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
    url = STATSAPI_GAME_CONTENT_URL.format(game_pk=game_pk)
    payload = _fetch_json(url, timeout_sec=timeout_sec)
    if cache_dir is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def _walk_json(value: Any, path: str = "$") -> Iterable[tuple[str, Any]]:
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_json(child, f"{path}[{index}]")


def _best_playback_url(playbacks: Any) -> str | None:
    if not isinstance(playbacks, list):
        return None
    candidates = []
    for playback in playbacks:
        if not isinstance(playback, dict):
            continue
        url = playback.get("url") or playback.get("href")
        if not url:
            continue
        name = str(playback.get("name") or playback.get("width") or "")
        candidates.append((str(url), name))
    if not candidates:
        return None
    mp4 = [item for item in candidates if ".mp4" in item[0].lower()]
    if mp4:
        return mp4[-1][0]
    hls = [item for item in candidates if ".m3u8" in item[0].lower()]
    if hls:
        return hls[-1][0]
    return candidates[-1][0]


def _extract_media_urls_from_text(text: str) -> list[str]:
    """Extract direct video URLs from Baseball Savant HTML/JSON text."""

    normalized_text = text.replace("\\\\/", "/").replace("\\/", "/")
    urls = re.findall(r"https?://[^\"'\s<>]+?(?:\.mp4|\.m3u8)(?:\?[^\"'\s<>]+)?", normalized_text)
    cleaned = []
    seen = set()
    for url in urls:
        if url not in seen:
            seen.add(url)
            cleaned.append(url)
    return cleaned


def _best_media_url_from_text(text: str) -> str | None:
    urls = _extract_media_urls_from_text(text)
    mp4 = [url for url in urls if ".mp4" in url.lower()]
    if mp4:
        return mp4[-1]
    hls = [url for url in urls if ".m3u8" in url.lower()]
    if hls:
        return hls[-1]
    return None


def _stable_asset_id(asset: dict[str, Any], raw_path: str) -> str:
    for key in ("id", "guid", "mediaPlaybackId", "slug"):
        value = asset.get(key)
        if value:
            return str(value)
    digest = hashlib.sha1((raw_path + json.dumps(asset, sort_keys=True, default=str))[:2000].encode()).hexdigest()[:12]
    return f"statsapi_asset_{digest}"


def extract_video_assets(game_content: dict[str, Any]) -> list[VideoAsset]:
    """Extract media/video-like assets from a generic MLB content payload."""

    assets: list[VideoAsset] = []
    seen: set[str] = set()
    for raw_path, value in _walk_json(game_content):
        if not isinstance(value, dict):
            continue
        title = str(value.get("title") or value.get("headline") or value.get("blurb") or "")
        description = str(value.get("description") or value.get("summary") or value.get("seoTitle") or "")
        playbacks = value.get("playbacks") or value.get("playback") or value.get("mediaPlayback")
        has_playbacks = isinstance(playbacks, list)
        media_url = _best_playback_url(playbacks)
        direct_url = value.get("url") if isinstance(value.get("url"), str) else None
        if media_url is None and direct_url and any(suffix in direct_url.lower() for suffix in (".mp4", ".m3u8")):
            media_url = direct_url
        if not has_playbacks and not title:
            continue
        if not media_url and not title:
            continue
        if not media_url and "video" not in raw_path.lower():
            continue
        asset_id = _stable_asset_id(value, raw_path)
        key = f"{asset_id}:{media_url}"
        if key in seen:
            continue
        seen.add(key)
        duration = value.get("duration") or value.get("durationSeconds")
        try:
            duration_sec = float(duration) if duration is not None else None
        except (TypeError, ValueError):
            duration_sec = None
        assets.append(
            VideoAsset(
                source_video_id=asset_id,
                title=title,
                description=description,
                source_url=str(value.get("url") or value.get("shareUrl") or "") or None,
                media_url=media_url,
                duration_sec=duration_sec,
                published_date=str(value.get("date") or value.get("updated") or "") or None,
                raw_path=raw_path,
            )
        )
    return assets


def resolve_baseball_savant_play_media(
    play_id: str,
    *,
    cache_dir: str | Path | None = None,
    timeout_sec: int = 60,
) -> tuple[str, str | None]:
    """Resolve a Baseball Savant play page into a direct media URL when exposed.

    Baseball Savant's public page shape has changed over time. The resolver keeps
    the source page even when a direct mp4/HLS URL is not visible, so downstream
    reports can separate "referenced page exists" from "downloadable media URL".
    """

    url = BASEBALL_SAVANT_SPORTY_URL.format(play_id=play_id)
    text = ""
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / "baseball_savant" / f"sporty_{play_id}.html"
        if cache_path.exists():
            text = cache_path.read_text(encoding="utf-8", errors="replace")
    if not text:
        try:
            text = _fetch_text(url, timeout_sec=timeout_sec)
        except Exception:
            return url, None
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(text, encoding="utf-8")
    return url, _best_media_url_from_text(text)


def build_baseball_savant_play_video_source_row(
    event: dict[str, Any],
    *,
    media_url: str | None,
    source_url: str,
    candidate_rank: int,
) -> dict[str, Any]:
    event_id = str(event["event_id"])
    play_id = str(event.get("play_id") or "")
    return {
        "schema_version": VIDEO_SOURCES_SCHEMA.version,
        "video_source_id": f"vs_{event_id}_savant_play_{candidate_rank}",
        "event_id": event_id,
        "same_event_group_id": str(event.get("same_event_group_id", event_id)),
        "source_video_id": play_id or None,
        "view_id": f"{event_id}_savant_play_view{candidate_rank}",
        "source_kind": "baseball_savant_sporty",
        "source_url": source_url,
        "media_url": media_url,
        "source_topic": "baseball_savant_play",
        "dataset_role": str(event.get("dataset_role", "train_candidate")),
        "rights_status": "official_public_reference",
        "match_confidence": 0.70 if media_url else 0.45,
        "match_reason": "exact_play_id_sporty_video" if media_url else "exact_play_id_sporty_page_no_direct_media",
        "join_key_fields": ["play_id", "game_pk", "event_id"],
        "candidate_rank": candidate_rank,
        "video_available": bool(media_url or source_url),
        "download_status": "not_attempted" if media_url else "referenced_only",
        "local_video_path": None,
        "probe_status": "pending",
        "review_status": "pending",
        "reject_reason": None,
        "view_label": "broadcast" if media_url else "unknown",
        "view_confidence": 0.65 if media_url else 0.0,
        "batting_visibility": "unknown",
        "is_replay": False,
        "is_non_batting_segment": False,
    }


def _event_match_score(event: dict[str, Any], asset: VideoAsset) -> tuple[float, str]:
    text = f"{asset.title} {asset.description}".lower()
    score = 0.20
    reasons = ["same_game_content"]
    player_name = str(event.get("player_name") or "").lower()
    if player_name and player_name != "unknown":
        parts = [part for part in player_name.replace(",", " ").split() if len(part) >= 3]
        hits = sum(1 for part in parts if part in text)
        if hits:
            score += min(0.35, 0.15 * hits)
            reasons.append("player_name_text_match")
    events = str(event.get("events") or "").replace("_", " ").lower()
    if events and events in text:
        score += 0.20
        reasons.append("event_text_match")
    description = str(event.get("description") or "").replace("_", " ").lower()
    if description and description in text:
        score += 0.10
        reasons.append("description_text_match")
    if asset.media_url:
        score += 0.10
        reasons.append("direct_media_url_present")
    return min(score, 0.99), "+".join(reasons)


def _inferred_view_from_match(score: float, reason: str, is_replay: bool) -> tuple[str, float]:
    """Infer a conservative broadcast view only for event-level match evidence."""

    if is_replay:
        return "unknown", 0.0
    if score < 0.45 or not has_event_level_match_reason(reason):
        return "unknown", 0.0
    return "broadcast", min(0.75, max(0.55, float(score)))


def _has_event_level_downloadable(row: dict[str, Any]) -> bool:
    return bool(row.get("media_url")) and has_event_level_match_reason(str(row.get("match_reason") or ""))


def _source_topic(event: dict[str, Any]) -> str:
    if event.get("is_home_run"):
        return "home_runs"
    return "statsapi_game_content"


def build_mlb_statsapi_video_source_rows(
    bbe_rows: list[dict[str, Any]],
    *,
    cache_dir: str | Path | None = None,
    max_assets_per_event: int = 2,
    min_match_confidence: float = 0.45,
    timeout_sec: int = 60,
    fallback_probe_rows: bool = True,
    include_savant_play_resolver: bool = True,
    max_events: int | None = None,
    max_savant_play_resolves: int | None = None,
    output_path: str | Path | None = None,
    progress_path: str | Path | None = None,
    resume: bool = True,
    checkpoint_every_events: int = 1,
) -> list[dict[str, Any]]:
    """Resolve game-content assets into `video_sources_v1` rows."""

    content_by_game: dict[int, dict[str, Any]] = {}
    assets_by_game: dict[int, list[VideoAsset]] = {}
    output = Path(output_path) if output_path is not None else None
    progress = Path(progress_path) if progress_path is not None else None
    rows: list[dict[str, Any]] = read_table(output) if resume and output is not None and output.exists() else []
    progress_payload: dict[str, Any] = {}
    if resume and progress is not None and progress.exists():
        try:
            progress_payload = json.loads(progress.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            progress_payload = {}
    processed_event_ids = {
        str(row.get("event_id"))
        for row in rows
        if row.get("event_id") is not None
    } | {str(event_id) for event_id in progress_payload.get("completed_event_ids", [])}
    failed_events: list[dict[str, Any]] = []
    savant_resolves = 0
    events_to_process = bbe_rows[:max_events] if max_events is not None else bbe_rows
    total_events = len(events_to_process)

    def write_progress(completed_events: int, status: str) -> None:
        if progress is None:
            return
        payload = {
            "schema_version": "statsapi_video_resolve_progress_v1",
            "status": status,
            "total_events": total_events,
            "completed_events": completed_events,
            "rows": len(rows),
            "media_url_rows": sum(1 for row in rows if row.get("media_url")),
            "completed_event_ids": sorted(processed_event_ids),
            "resume": resume,
            "output_path": str(output) if output is not None else None,
            "failed_events": failed_events[-100:],
            "max_events": max_events,
            "max_savant_play_resolves": max_savant_play_resolves,
        }
        progress.parent.mkdir(parents=True, exist_ok=True)
        progress.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def checkpoint(completed_events: int, status: str) -> None:
        if output is not None:
            write_table(output, rows)
        write_progress(completed_events, status)

    for completed_events, event in enumerate(events_to_process, start=1):
        event_id_for_resume = str(event.get("event_id"))
        if resume and event_id_for_resume in processed_event_ids:
            if completed_events % max(1, checkpoint_every_events) == 0:
                write_progress(completed_events, "running_cached")
            continue
        try:
            game_pk = int(event["game_pk"])
            if game_pk not in content_by_game:
                content_by_game[game_pk] = fetch_game_content(game_pk, cache_dir=cache_dir, timeout_sec=timeout_sec)
                assets_by_game[game_pk] = extract_video_assets(content_by_game[game_pk])
            scored = []
            for asset in assets_by_game[game_pk]:
                score, reason = _event_match_score(event, asset)
                if score >= min_match_confidence:
                    scored.append((score, reason, asset))
            scored.sort(key=lambda item: item[0], reverse=True)
            for rank, (score, reason, asset) in enumerate(scored[:max_assets_per_event], start=1):
                event_id = str(event["event_id"])
                source_topic = _source_topic(event)
                dataset_role = "smoke_test" if source_topic == "home_runs" else str(event.get("dataset_role", "train_candidate"))
                is_replay = "replay" in f"{asset.title} {asset.description}".lower()
                view_label, view_confidence = _inferred_view_from_match(score, reason, is_replay)
                rows.append(
                    {
                        "schema_version": VIDEO_SOURCES_SCHEMA.version,
                        "video_source_id": f"vs_{event_id}_statsapi_{rank}",
                        "event_id": event_id,
                        "same_event_group_id": str(event.get("same_event_group_id", event_id)),
                        "source_video_id": asset.source_video_id,
                        "view_id": f"{event_id}_statsapi_view{rank}",
                        "source_kind": "statsapi_content",
                        "source_url": asset.source_url or STATSAPI_GAME_CONTENT_URL.format(game_pk=game_pk),
                        "media_url": asset.media_url,
                        "source_topic": source_topic,
                        "dataset_role": dataset_role,
                        "rights_status": "official_public_reference",
                        "match_confidence": float(score),
                        "match_reason": reason,
                        "join_key_fields": ["game_pk", "player_name", "events", "description"],
                        "candidate_rank": rank,
                        "video_available": bool(asset.media_url or asset.source_url),
                        "download_status": "not_attempted" if asset.media_url else "referenced_only",
                        "local_video_path": None,
                        "probe_status": "pending",
                        "review_status": "pending",
                        "reject_reason": None,
                        "view_label": view_label,
                        "view_confidence": view_confidence,
                        "batting_visibility": "unknown",
                        "is_replay": is_replay,
                        "is_non_batting_segment": False,
                    }
                )
            can_resolve_savant = (
                include_savant_play_resolver
                and event.get("play_id")
                and (max_savant_play_resolves is None or savant_resolves < max_savant_play_resolves)
            )
            if can_resolve_savant:
                event_rows = [row for row in rows if row["event_id"] == str(event["event_id"])]
                has_downloadable = any(_has_event_level_downloadable(row) for row in event_rows)
                if not has_downloadable:
                    savant_resolves += 1
                    source_url, media_url = resolve_baseball_savant_play_media(
                        str(event["play_id"]),
                        cache_dir=cache_dir,
                        timeout_sec=timeout_sec,
                    )
                    rows.append(
                        build_baseball_savant_play_video_source_row(
                            event,
                            media_url=media_url,
                            source_url=source_url,
                            candidate_rank=len(event_rows) + 1,
                        )
                    )
            processed_event_ids.add(event_id_for_resume)
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            failed_events.append({"event_id": event_id_for_resume, "game_pk": event.get("game_pk"), "error": str(exc)})
        if completed_events % max(1, checkpoint_every_events) == 0:
            checkpoint(completed_events, "running")
    if not rows and fallback_probe_rows:
        rows = build_video_source_rows(bbe_rows)
    validate_rows(VIDEO_SOURCES_SCHEMA, rows)
    checkpoint(total_events, "complete")
    return rows


def build_mlb_statsapi_video_source_artifact(
    base_dir: str | Path,
    bbe_path: str | Path,
    *,
    cache_dir: str | Path | None = None,
    max_assets_per_event: int = 2,
    min_match_confidence: float = 0.45,
    output_suffix: str = ".parquet",
    timeout_sec: int = 60,
    include_savant_play_resolver: bool = True,
    max_events: int | None = None,
    max_savant_play_resolves: int | None = None,
    resume: bool = True,
    checkpoint_every_events: int = 1,
) -> Path:
    bbe_rows = read_table(bbe_path)
    output = Path(base_dir) / f"manifests/video_sources_v1{output_suffix}"
    progress = Path(base_dir) / "reports/preflight/statsapi_video_resolve_progress_v1.json"
    rows = build_mlb_statsapi_video_source_rows(
        bbe_rows,
        cache_dir=cache_dir,
        max_assets_per_event=max_assets_per_event,
        min_match_confidence=min_match_confidence,
        timeout_sec=timeout_sec,
        include_savant_play_resolver=include_savant_play_resolver,
        max_events=max_events,
        max_savant_play_resolves=max_savant_play_resolves,
        output_path=output,
        progress_path=progress,
        resume=resume,
        checkpoint_every_events=checkpoint_every_events,
    )
    write_table(output, rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve MLB Stats API game-content video candidates.")
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/baseball_vision")
    parser.add_argument("--bbe-events", default=None)
    parser.add_argument("--cache-dir", default="/content/cache/baseball_vision/statsapi")
    parser.add_argument("--max-assets-per-event", type=int, default=2)
    parser.add_argument("--min-match-confidence", type=float, default=0.45)
    parser.add_argument("--output-format", choices=("parquet", "jsonl", "json", "csv"), default="parquet")
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--disable-savant-play-resolver", action="store_true")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--max-savant-play-resolves", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--checkpoint-every-events", type=int, default=1)
    args = parser.parse_args()
    bbe_path = Path(args.bbe_events) if args.bbe_events else Path(args.base_dir) / "manifests/bbe_events_v1.parquet"
    output = build_mlb_statsapi_video_source_artifact(
        base_dir=args.base_dir,
        bbe_path=bbe_path,
        cache_dir=args.cache_dir,
        max_assets_per_event=args.max_assets_per_event,
        min_match_confidence=args.min_match_confidence,
        output_suffix="." + args.output_format,
        timeout_sec=args.timeout_sec,
        include_savant_play_resolver=not args.disable_savant_play_resolver,
        max_events=args.max_events,
        max_savant_play_resolves=args.max_savant_play_resolves,
        resume=not args.no_resume,
        checkpoint_every_events=args.checkpoint_every_events,
    )
    print(json.dumps({"video_sources": str(output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
