"""Download MLB player-season hitting totals for BA/OPS/OBP/SLG labels.

This module uses the public MLB StatsAPI season hitting endpoint. It writes
player-season labels that can be joined to Statcast batter ids / seasons
without treating BA, OPS, OBP, or SLG as event-level BBE targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable
import urllib.parse
import urllib.request

from sport_pipeline.artifact_check import write_json
from sport_pipeline.io import read_table, write_table


STATSAPI_BASE = "https://statsapi.mlb.com/api/v1/stats"


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() in {"", "-", ".---"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() in {"", "-"}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _season_from_bbe_events(base_dir: Path) -> list[int]:
    bbe_path = base_dir / "manifests/bbe_events_v1.parquet"
    if not bbe_path.exists():
        bbe_path = base_dir / "manifests/bbe_events_v1.jsonl"
    if not bbe_path.exists():
        return []
    seasons: set[int] = set()
    for row in read_table(bbe_path):
        value = row.get("season") or str(row.get("game_date", ""))[:4]
        try:
            seasons.add(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(seasons)


def _statsapi_url(season: int, *, sport_id: int = 1, hydrate: str | None = None) -> str:
    query = {
        "stats": "season",
        "group": "hitting",
        "playerPool": "all",
        "season": str(season),
        "sportIds": str(sport_id),
        "limit": "10000",
    }
    if hydrate:
        query["hydrate"] = hydrate
    return f"{STATSAPI_BASE}?{urllib.parse.urlencode(query)}"


def _fetch_json(url: str, *, timeout_sec: int) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "sport-pipeline-research/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        return json.loads(response.read().decode("utf-8"))


def rows_from_statsapi_payload(payload: dict[str, Any], *, season: int) -> list[dict[str, Any]]:
    """Normalize one MLB StatsAPI hitting stats payload."""

    rows: list[dict[str, Any]] = []
    for block in payload.get("stats", []):
        for split in block.get("splits", []):
            player = split.get("player") or {}
            stat = split.get("stat") or {}
            team = split.get("team") or {}
            batter_id = player.get("id")
            if batter_id is None:
                continue
            batter_id_text = str(batter_id)
            row = {
                "schema_version": "player_season_batting_stats_v1",
                "source": "mlb_statsapi",
                "batter_id": batter_id_text,
                "batter_name": player.get("fullName"),
                "season": int(season),
                "batter_season_id": f"{batter_id_text}_{int(season)}",
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "games_played": _to_int(stat.get("gamesPlayed")),
                "plate_appearances": _to_int(stat.get("plateAppearances")),
                "at_bats": _to_int(stat.get("atBats")),
                "hits": _to_int(stat.get("hits")),
                "doubles": _to_int(stat.get("doubles")),
                "triples": _to_int(stat.get("triples")),
                "home_runs": _to_int(stat.get("homeRuns")),
                "walks": _to_int(stat.get("baseOnBalls")),
                "intentional_walks": _to_int(stat.get("intentionalWalks")),
                "strikeouts": _to_int(stat.get("strikeOuts")),
                "hit_by_pitch": _to_int(stat.get("hitByPitch")),
                "sac_flies": _to_int(stat.get("sacFlies")),
                "total_bases": _to_int(stat.get("totalBases")),
                "target_ba": _to_float(stat.get("avg")),
                "target_obp": _to_float(stat.get("obp")),
                "target_slg": _to_float(stat.get("slg")),
                "target_ops": _to_float(stat.get("ops")),
            }
            row["avg"] = row["target_ba"]
            for target in ("ba", "ops", "obp", "slg"):
                available = row.get(f"target_{target}") is not None
                row[f"target_{target}_available"] = available
                row[f"target_{target}_missing_reason"] = None if available else "mlb_statsapi_stat_missing"
            rows.append(row)
    return rows


def _dedupe_player_seasons(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("batter_season_id") or "")
        if not key:
            continue
        current = by_key.get(key)
        if current is None:
            by_key[key] = row
            continue
        current_pa = _to_int(current.get("plate_appearances")) or -1
        row_pa = _to_int(row.get("plate_appearances")) or -1
        if row_pa >= current_pa:
            by_key[key] = row
    return [by_key[key] for key in sorted(by_key)]


def download_player_season_batting_stats(
    base_dir: str | Path,
    *,
    seasons: Iterable[int] | None = None,
    execute: bool = True,
    sport_id: int = 1,
    timeout_sec: int = 60,
    force: bool = False,
    output_suffix: str = ".parquet",
) -> dict[str, Path]:
    """Download annual player-season hitting totals and write label manifest."""

    base = Path(base_dir)
    resolved_seasons = sorted({int(season) for season in seasons}) if seasons is not None else _season_from_bbe_events(base)
    outputs = {
        "manifest": base / f"manifests/player_season_batting_v1{output_suffix}",
        "summary": base / "reports/preflight/player_season_batting_stats_v1.json",
    }
    raw_dir = base / "raw_player_stats"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    season_summaries: list[dict[str, Any]] = []
    for season in resolved_seasons:
        raw_path = raw_dir / f"mlb_statsapi_hitting_{season}.json"
        status = "planned"
        payload: dict[str, Any] | None = None
        if execute:
            if raw_path.exists() and not force:
                payload = json.loads(raw_path.read_text(encoding="utf-8"))
                status = "reused_raw_json"
            else:
                url = _statsapi_url(season, sport_id=sport_id)
                payload = _fetch_json(url, timeout_sec=timeout_sec)
                raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                status = "downloaded"
            rows = rows_from_statsapi_payload(payload, season=season)
            all_rows.extend(rows)
            season_summaries.append({"season": season, "status": status, "rows": len(rows), "raw_path": str(raw_path)})
        else:
            season_summaries.append(
                {
                    "season": season,
                    "status": status,
                    "url": _statsapi_url(season, sport_id=sport_id),
                    "raw_path": str(raw_path),
                }
            )

    rows_to_write = _dedupe_player_seasons(all_rows)
    if execute:
        write_table(outputs["manifest"], rows_to_write)
    write_json(
        {
            "schema_version": "player_season_batting_stats_summary_v1",
            "execute": execute,
            "seasons": resolved_seasons,
            "sport_id": sport_id,
            "season_summaries": season_summaries,
            "manifest_rows": len(rows_to_write),
            "ba_rows": sum(1 for row in rows_to_write if row.get("target_ba") is not None),
            "ops_rows": sum(1 for row in rows_to_write if row.get("target_ops") is not None),
            "obp_rows": sum(1 for row in rows_to_write if row.get("target_obp") is not None),
            "slg_rows": sum(1 for row in rows_to_write if row.get("target_slg") is not None),
            "outputs": {key: str(path) for key, path in outputs.items()},
        },
        outputs["summary"],
    )
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download player-season hitting totals for BA/OPS/OBP/SLG labels.")
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/baseball_vision")
    parser.add_argument("--seasons", default=None, help="Comma-separated seasons. Defaults to seasons found in bbe_events.")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--sport-id", type=int, default=1)
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-format", choices=("parquet", "jsonl", "json", "csv"), default="parquet")
    args = parser.parse_args(argv)
    seasons = None
    if args.seasons:
        seasons = [int(part.strip()) for part in args.seasons.split(",") if part.strip()]
    outputs = download_player_season_batting_stats(
        args.base_dir,
        seasons=seasons,
        execute=not args.plan_only,
        sport_id=args.sport_id,
        timeout_sec=args.timeout_sec,
        force=args.force,
        output_suffix="." + args.output_format,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
