"""Build BBE manifest and split artifacts from Statcast-style rows."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sport_pipeline.io import read_table, write_table
from sport_pipeline.io.jsonl import read_jsonl
from sport_pipeline.manifests import build_batter_season_id, build_event_id, build_same_event_group_id
from sport_pipeline.schemas import BBE_EVENTS_SCHEMA, PLAYER_GROUP_SPLIT_SCHEMA, TEMPORAL_SPLIT_SCHEMA
from sport_pipeline.schemas.data_manifest import validate_rows
from sport_pipeline.statcast.targets import ops_missing_reason


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_BBE_EVENTS = PROJECT_ROOT / "manifests/templates/bbe_events_v1.sample.jsonl"


def _none_if_blank(value: Any) -> Any:
    if value == "" or value == "null":
        return None
    return value


def _int(value: Any) -> int | None:
    value = _none_if_blank(value)
    if value is None:
        return None
    return int(float(value))


def _float(value: Any) -> float | None:
    value = _none_if_blank(value)
    if value is None:
        return None
    return float(value)


def _str(value: Any, default: str = "") -> str:
    value = _none_if_blank(value)
    if value is None:
        return default
    return str(value)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _year_from_date(game_date: str | None) -> int | None:
    if not game_date:
        return None
    return int(str(game_date)[:4])


BATTER_NAME_VERBS = (
    "singles",
    "doubles",
    "triples",
    "homers",
    "grounds",
    "flies",
    "lines",
    "pops",
    "reaches",
    "hits",
    "bunts",
    "sacrifices",
    "walks",
    "strikes",
    "called",
    "out",
)


def _batter_name_from_des(description: Any) -> str | None:
    """Infer the batter display name from Statcast `des` play text.

    Baseball Savant CSVs fetched with `player_type=pitcher` set
    `player_name` to the pitcher. For video matching we need the batter name,
    which is usually the leading phrase in `des`.
    """

    text = _str(description).strip()
    if not text:
        return None
    pattern = r"^(.+?)\s+(?:" + "|".join(BATTER_NAME_VERBS) + r")\b"
    match = re.match(pattern, text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip(" ,.")
    first_sentence = text.split(". ", 1)[0].strip(" ,.")
    return first_sentence or None


def _outcome_bin(events: str) -> str:
    if events == "home_run":
        return "home_run"
    if events == "single":
        return "single"
    if events in {"double", "triple"}:
        return "double_triple"
    if "error" in events or "fielders_choice" in events:
        return "error_or_reach"
    return "out"


def _ev_bin(ev: float | None) -> str:
    if ev is None:
        return "ev_missing"
    if ev < 80:
        return "ev_lt_80"
    if ev < 90:
        return "ev_80_90"
    if ev < 95:
        return "ev_90_95"
    if ev < 100:
        return "ev_95_100"
    return "ev_ge_100"


def _la_bin(la: float | None) -> str:
    if la is None:
        return "la_missing"
    if la < 0:
        return "ground"
    if la < 8:
        return "low"
    if la <= 32:
        return "sweet_spot"
    if la <= 50:
        return "high_fly"
    return "popup"


def _barrel_label(row: dict[str, Any], ev: float | None, la: float | None) -> float | None:
    if row.get("target_barrel") is not None:
        return float(row["target_barrel"])
    speed_angle = _int(row.get("launch_speed_angle"))
    if speed_angle is not None:
        return 1.0 if speed_angle == 6 else 0.0
    if ev is None or la is None:
        return None
    return 1.0 if ev >= 98.0 and 26.0 <= la <= 30.0 else 0.0


def normalize_bbe_row(raw: dict[str, Any], dataset_role: str = "train_candidate") -> dict[str, Any]:
    """Normalize a Statcast-style row into bbe_events_v1."""

    if raw.get("schema_version") == BBE_EVENTS_SCHEMA.version:
        row = dict(raw)
        ev = _float(row.get("launch_speed"))
        la = _float(row.get("launch_angle"))
        row.setdefault("target_hard_hit", None if ev is None else float(ev >= 95.0))
        row.setdefault("target_barrel", _barrel_label(row, ev, la))
        return row

    game_pk = _int(raw.get("game_pk"))
    game_date = _str(raw.get("game_date") or raw.get("game_date_dt"))
    season = _int(raw.get("season")) or _year_from_date(game_date)
    batter_id = _int(raw.get("batter_id") or raw.get("batter"))
    pitcher_id = _int(raw.get("pitcher_id") or raw.get("pitcher"))
    at_bat_number = _int(raw.get("at_bat_number"))
    pitch_number = _int(raw.get("pitch_number"))
    play_id = _none_if_blank(raw.get("play_id"))
    sv_id = _none_if_blank(raw.get("sv_id"))
    if game_pk is None or season is None or batter_id is None or pitcher_id is None:
        raise ValueError("game_pk, season/game_date, batter, and pitcher are required")
    event_id = _str(raw.get("event_id")) or build_event_id(game_pk, at_bat_number, pitch_number, sv_id=sv_id, play_id=play_id)
    batter_season_id = _str(raw.get("batter_season_id")) or build_batter_season_id(batter_id, season)
    ev = _float(raw.get("launch_speed"))
    la = _float(raw.get("launch_angle"))
    xba = _float(raw.get("estimated_ba_using_speedangle"))
    xwoba = _float(raw.get("estimated_woba_using_speedangle"))
    events = _str(raw.get("events"), "unknown")
    zone = _int(raw.get("zone"))
    strikes = _int(raw.get("strikes"))
    has_video = _bool(raw.get("has_video_candidate")) if "has_video_candidate" in raw else bool(raw.get("source_url") or raw.get("media_url"))
    target_barrel = _barrel_label(raw, ev, la)
    play_description = _str(raw.get("des") or raw.get("description"), "")
    batter_name = _str(raw.get("batter_name") or _batter_name_from_des(raw.get("des")) or raw.get("player_name"), "unknown")
    target_ev_available = ev is not None
    target_la_available = la is not None
    optional_missing = xba is None or xwoba is None
    label_missing_reason = None
    if not target_ev_available or not target_la_available:
        label_missing_reason = "launch_metric_missing"
    elif optional_missing:
        label_missing_reason = "statcast_expected_outcome_missing"

    return {
        "schema_version": BBE_EVENTS_SCHEMA.version,
        "event_id": event_id,
        "game_pk": game_pk,
        "game_date": game_date,
        "season": season,
        "batter_id": batter_id,
        "pitcher_id": pitcher_id,
        "batter_season_id": batter_season_id,
        "at_bat_number": at_bat_number,
        "pitch_number": pitch_number,
        "play_id": play_id,
        "same_event_group_id": _str(raw.get("same_event_group_id")) or build_same_event_group_id(event_id),
        "player_name": batter_name,
        "events": events,
        "description": play_description,
        "bb_type": _none_if_blank(raw.get("bb_type")),
        "launch_speed": ev,
        "launch_angle": la,
        "launch_speed_angle": _int(raw.get("launch_speed_angle")),
        "estimated_ba_using_speedangle": xba,
        "estimated_woba_using_speedangle": xwoba,
        "stand": _none_if_blank(raw.get("stand")),
        "p_throws": _none_if_blank(raw.get("p_throws")),
        "pitch_type": _none_if_blank(raw.get("pitch_type")),
        "release_speed": _float(raw.get("release_speed")),
        "plate_x": _float(raw.get("plate_x")),
        "plate_z": _float(raw.get("plate_z")),
        "zone": zone,
        "balls": _int(raw.get("balls")),
        "strikes": strikes,
        "outs_when_up": _int(raw.get("outs_when_up")),
        "inning": _int(raw.get("inning")),
        "inning_topbot": _none_if_blank(raw.get("inning_topbot")),
        "home_team": _str(raw.get("home_team"), "unknown"),
        "away_team": _str(raw.get("away_team"), "unknown"),
        "sv_id": sv_id,
        "is_bbe": True,
        "is_home_run": events == "home_run",
        "dataset_role": dataset_role,
        "outcome_bin": _outcome_bin(events),
        "ev_bin": _ev_bin(ev),
        "la_bin": _la_bin(la),
        "bb_type_bin": _none_if_blank(raw.get("bb_type")) or "unknown",
        "has_video_candidate": has_video,
        "n_video_candidates": _int(raw.get("n_video_candidates")) or (1 if has_video else 0),
        "video_availability_score": _float(raw.get("video_availability_score")) or (0.5 if has_video else 0.0),
        "target_ev_available": target_ev_available,
        "target_la_available": target_la_available,
        "target_hard_hit_available": ev is not None,
        "target_barrel_available": target_barrel is not None,
        "target_xba_available": xba is not None,
        "target_xwoba_available": xwoba is not None,
        "target_ops_available": False,
        "target_ops_missing_reason": ops_missing_reason(False),
        "label_missing_reason": label_missing_reason,
        "clean_location_cohort_v1": zone in {2, 4, 5, 6, 8} if zone is not None else False,
        "clean_count_cohort_v1": strikes is not None and strikes <= 1,
        "usable_for_event_model": target_ev_available and target_la_available,
        "quality_flags": [] if has_video else ["no_video_candidate"],
        "outlier_flags": [],
        "review_status": "usable_primary" if target_ev_available and target_la_available else "pending",
        "reject_reason": None,
        "target_hard_hit": None if ev is None else float(ev >= 95.0),
        "target_barrel": target_barrel,
    }


def build_bbe_manifest_rows(input_path: str | Path | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    """Build normalized BBE manifest rows."""

    raw_rows = read_jsonl(SAMPLE_BBE_EVENTS) if input_path is None else read_table(input_path)
    rows = [normalize_bbe_row(row) for row in raw_rows]
    if limit is not None:
        rows = rows[:limit]
    validate_rows(BBE_EVENTS_SCHEMA, rows)
    return rows


def build_split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build deterministic player-group and temporal split rows."""

    created_at = datetime.now(timezone.utc).isoformat()
    groups = sorted({row["batter_season_id"] for row in rows})
    group_to_split: dict[str, str] = {}
    for index, group in enumerate(groups):
        frac = (index + 1) / max(len(groups), 1)
        if frac <= 0.70:
            split = "train"
        elif frac <= 0.85:
            split = "validation"
        else:
            split = "test"
        group_to_split[group] = split
    player_rows = [
        {
            "schema_version": PLAYER_GROUP_SPLIT_SCHEMA.version,
            "event_id": row["event_id"],
            "batter_id": row["batter_id"],
            "season": row["season"],
            "batter_season_id": row["batter_season_id"],
            "split": group_to_split[row["batter_season_id"]],
            "split_strategy": "player_group_v1",
            "group_key": row["batter_season_id"],
            "created_at": created_at,
        }
        for row in rows
    ]

    sorted_rows = sorted(rows, key=lambda row: (row["game_date"], row["event_id"]))
    temporal_by_event: dict[str, str] = {}
    for index, row in enumerate(sorted_rows):
        frac = (index + 1) / max(len(sorted_rows), 1)
        if frac <= 0.70:
            split = "train"
        elif frac <= 0.85:
            split = "validation"
        else:
            split = "test"
        temporal_by_event[row["event_id"]] = split
    cutoff_index = max(0, min(len(sorted_rows) - 1, int(len(sorted_rows) * 0.70))) if sorted_rows else 0
    cutoff_date = sorted_rows[cutoff_index]["game_date"] if sorted_rows else ""
    temporal_rows = [
        {
            "schema_version": TEMPORAL_SPLIT_SCHEMA.version,
            "event_id": row["event_id"],
            "batter_id": row["batter_id"],
            "season": row["season"],
            "batter_season_id": row["batter_season_id"],
            "game_date": row["game_date"],
            "split": temporal_by_event[row["event_id"]],
            "split_strategy": "temporal_v1",
            "cutoff_date": cutoff_date,
            "created_at": created_at,
        }
        for row in rows
    ]
    validate_rows(PLAYER_GROUP_SPLIT_SCHEMA, player_rows)
    validate_rows(TEMPORAL_SPLIT_SCHEMA, temporal_rows)
    return player_rows, temporal_rows


def build_manifest_artifacts(
    base_dir: str | Path,
    input_path: str | Path | None = None,
    *,
    limit: int | None = None,
    output_suffix: str = ".parquet",
) -> dict[str, Path]:
    """Write BBE manifest and split artifacts under BASE_DIR."""

    base = Path(base_dir)
    rows = build_bbe_manifest_rows(input_path, limit=limit)
    player_rows, temporal_rows = build_split_rows(rows)
    outputs = {
        "bbe_events": base / f"manifests/bbe_events_v1{output_suffix}",
        "player_group_split": base / f"manifests/splits/player_group_split_v1{output_suffix}",
        "temporal_split": base / f"manifests/splits/temporal_split_v1{output_suffix}",
    }
    write_table(outputs["bbe_events"], rows)
    write_table(outputs["player_group_split"], player_rows)
    write_table(outputs["temporal_split"], temporal_rows)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build bbe_events_v1 and split artifacts.")
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/baseball_vision")
    parser.add_argument("--input", default=None, help="Optional Statcast CSV/JSONL/JSON/Parquet input. Defaults to bundled smoke sample.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-format", choices=("parquet", "jsonl", "json", "csv"), default="parquet")
    args = parser.parse_args(argv)
    suffix = "." + args.output_format
    outputs = build_manifest_artifacts(args.base_dir, args.input, limit=args.limit, output_suffix=suffix)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
