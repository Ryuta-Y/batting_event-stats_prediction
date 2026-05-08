"""Build a lightweight context_dataset_v1 manifest from BBE rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sport_pipeline.io import read_table, write_table


CONTEXT_COLUMNS = (
    "event_id",
    "batter_season_id",
    "game_date",
    "season",
    "batter_id",
    "pitcher_id",
    "stand",
    "p_throws",
    "pitch_type",
    "release_speed",
    "plate_x",
    "plate_z",
    "zone",
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "inning_topbot",
    "launch_speed",
    "launch_angle",
    "target_hard_hit",
    "target_barrel",
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",
    "target_ops",
    "target_ops_missing_reason",
)


def build_context_dataset_rows(bbe_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select context baseline columns while preserving target metadata."""

    rows: list[dict[str, Any]] = []
    for row in bbe_rows:
        output = {column: row.get(column) for column in CONTEXT_COLUMNS if column in row}
        output["schema_version"] = "context_dataset_v1"
        output["sample_id"] = str(row.get("event_id"))
        output["prediction_level"] = "event"
        output["target_ev_available"] = bool(row.get("target_ev_available", row.get("launch_speed") is not None))
        output["target_la_available"] = bool(row.get("target_la_available", row.get("launch_angle") is not None))
        output["target_hard_hit_available"] = bool(row.get("target_hard_hit_available", row.get("target_hard_hit") is not None))
        output["target_barrel_available"] = bool(row.get("target_barrel_available", row.get("target_barrel") is not None))
        output["target_xba_available"] = bool(row.get("target_xba_available", row.get("estimated_ba_using_speedangle") is not None))
        output["target_xwoba_available"] = bool(row.get("target_xwoba_available", row.get("estimated_woba_using_speedangle") is not None))
        output["target_ops_available"] = bool(row.get("target_ops_available", False))
        output["label_missing_reason"] = row.get("label_missing_reason")
        rows.append(output)
    return rows


def build_context_dataset_artifact(
    base_dir: str | Path,
    bbe_path: str | Path,
    *,
    output_suffix: str = ".parquet",
) -> Path:
    rows = build_context_dataset_rows(read_table(bbe_path))
    output = Path(base_dir) / f"datasets/context_dataset_v1/manifest{output_suffix}"
    write_table(output, rows)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build context_dataset_v1 from bbe_events_v1.")
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/baseball_vision")
    parser.add_argument("--bbe-events", default=None)
    parser.add_argument("--output-format", choices=("parquet", "jsonl", "json", "csv"), default="parquet")
    args = parser.parse_args(argv)
    suffix = "." + args.output_format
    bbe_path = Path(args.bbe_events) if args.bbe_events else Path(args.base_dir) / "manifests/bbe_events_v1.parquet"
    output = build_context_dataset_artifact(args.base_dir, bbe_path, output_suffix=suffix)
    print(json.dumps({"context_dataset": str(output)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
