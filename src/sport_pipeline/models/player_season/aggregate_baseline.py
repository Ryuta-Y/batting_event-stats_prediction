"""Player-season aggregate baseline from mechanics-prior embeddings.

This runner predicts player-season BBE aggregates such as average EV,
hard-hit rate, and barrel rate. BA/OPS/OBP/SLG are enabled when an annual
player-season batting stats manifest is available.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from sport_pipeline.artifact_check import write_json
from sport_pipeline.evaluation import evaluate_predictions, load_target_registry, validate_prediction_rows
from sport_pipeline.io import read_table, write_table


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TARGET_REGISTRY = PROJECT_ROOT / "configs/targets/target_registry_v1.yaml"
PLAYER_SEASON_TARGETS = (
    "avg_ev",
    "avg_la",
    "avg_xba",
    "avg_xwoba",
    "hard_hit_rate",
    "barrel_rate",
    "ba",
    "ops",
    "obp",
    "slg",
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _to_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_optional_table(path: Path) -> list[dict[str, Any]]:
    return read_table(path) if path.exists() else []


def _split_map(base_dir: Path) -> dict[str, str]:
    for relative in (
        "manifests/splits/player_group_split_v1.parquet",
        "manifests/splits/temporal_split_v1.parquet",
        "manifests/splits/player_group_split_v1.jsonl",
        "manifests/splits/temporal_split_v1.jsonl",
    ):
        path = base_dir / relative
        if path.exists():
            return {str(row["event_id"]): str(row.get("split", "unknown")) for row in read_table(path)}
    return {}


def _append_value(values: dict[str, list[float]], target_name: str, value: Any) -> None:
    converted = _to_float(value)
    if converted is not None:
        values[target_name].append(converted)


def _aggregate_bbe_targets(event_rows: Iterable[dict[str, Any]], splits: dict[str, str]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        batter_season_id = row.get("batter_season_id")
        if batter_season_id is None:
            batter = row.get("batter_id") or row.get("batter")
            season = row.get("season") or str(row.get("game_date", ""))[:4]
            if batter is None or not season:
                continue
            batter_season_id = f"{batter}_{season}"
            row = dict(row)
            row["batter_season_id"] = batter_season_id
        grouped[str(batter_season_id)].append(row)

    aggregates: dict[str, dict[str, Any]] = {}
    for batter_season_id, rows in grouped.items():
        values: dict[str, list[float]] = defaultdict(list)
        split_counts = Counter(splits.get(str(row.get("event_id")), str(row.get("split", "unknown"))) for row in rows)
        for row in rows:
            _append_value(values, "avg_ev", row.get("launch_speed"))
            _append_value(values, "avg_la", row.get("launch_angle"))
            _append_value(values, "avg_xba", row.get("estimated_ba_using_speedangle"))
            _append_value(values, "avg_xwoba", row.get("estimated_woba_using_speedangle"))
            hard_hit = row.get("target_hard_hit")
            if _is_missing(hard_hit):
                launch_speed = _to_float(row.get("launch_speed"))
                hard_hit = None if launch_speed is None else float(launch_speed >= 95.0)
            _append_value(values, "hard_hit_rate", hard_hit)
            _append_value(values, "barrel_rate", row.get("target_barrel"))

        first = rows[0]
        aggregate = {
            "batter_season_id": batter_season_id,
            "batter_id": str(first.get("batter_id") or first.get("batter") or batter_season_id.split("_")[0]),
            "season": int(first.get("season") or str(first.get("game_date", "0"))[:4] or 0),
            "split": split_counts.most_common(1)[0][0] if split_counts else "unknown",
            "n_bbe": len(rows),
            "source_event_ids": [str(row.get("event_id")) for row in rows if row.get("event_id") is not None],
        }
        for target_name, target_values in values.items():
            aggregate[f"target_{target_name}"] = mean(target_values) if target_values else None
            aggregate[f"n_{target_name}_events"] = len(target_values)
        for target_name in ("ba", "ops", "obp", "slg"):
            aggregate[f"target_{target_name}"] = None
            aggregate[f"target_{target_name}_missing_reason"] = "pa_manifest_unavailable"
        aggregates[batter_season_id] = aggregate
    return aggregates


def _merge_player_season_batting_stats(
    aggregates: dict[str, dict[str, Any]],
    stats_rows: Iterable[dict[str, Any]],
) -> int:
    """Attach BA/OPS/OBP/SLG labels from annual player-season hitting totals."""

    attached = 0
    by_key = {str(row.get("batter_season_id") or ""): row for row in stats_rows if row.get("batter_season_id")}
    for batter_season_id, aggregate in aggregates.items():
        row = by_key.get(batter_season_id)
        if row is None:
            continue
        attached += 1
        for target_name in ("ba", "ops", "obp", "slg"):
            value = _to_float(row.get(f"target_{target_name}"))
            aggregate[f"target_{target_name}"] = value
            aggregate[f"target_{target_name}_missing_reason"] = None if value is not None else row.get(
                f"target_{target_name}_missing_reason"
            ) or "player_season_stat_missing"
            aggregate[f"target_{target_name}_source"] = "player_season_batting_stats"
        aggregate["plate_appearances"] = row.get("plate_appearances")
        aggregate["at_bats"] = row.get("at_bats")
        aggregate["player_season_batting_stats_source"] = row.get("source")
    return attached


def _latest_player_season_embeddings(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        batter_season_id = str(row.get("batter_season_id"))
        if not batter_season_id or batter_season_id == "None":
            continue
        current = latest.get(batter_season_id)
        if current is None:
            latest[batter_season_id] = row
            continue
        if str(row.get("cutoff_date", "")) >= str(current.get("cutoff_date", "")):
            latest[batter_season_id] = row
    return latest


def _mechanics_signal(row: dict[str, Any] | None, aggregate: dict[str, Any]) -> float:
    values = (row or {}).get("embedding_values") or []
    vector = [_to_float(value) for value in values if _to_float(value) is not None]
    embedding_signal = mean(vector) if vector else 0.0
    n_clips = _to_float((row or {}).get("n_clips_used")) or 0.0
    n_bbe = _to_float(aggregate.get("n_bbe")) or 0.0
    return float(embedding_signal + 0.01 * math.log1p(n_clips) + 0.001 * math.log1p(n_bbe))


def _fit_linear(samples: list[dict[str, Any]], target_column: str) -> dict[str, float] | None:
    train_rows = [row for row in samples if row.get("split") == "train" and not _is_missing(row.get(target_column))]
    if not train_rows:
        train_rows = [row for row in samples if not _is_missing(row.get(target_column))]
    if not train_rows:
        return None
    xs = [float(row["mechanics_signal"]) for row in train_rows]
    ys = [float(row[target_column]) for row in train_rows]
    mean_x = mean(xs)
    mean_y = mean(ys)
    var_x = sum((value - mean_x) ** 2 for value in xs)
    slope = 0.0 if var_x == 0 else sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / var_x
    return {"intercept": float(mean_y - slope * mean_x), "slope": float(slope), "train_rows": float(len(train_rows))}


def _build_samples(
    aggregates: dict[str, dict[str, Any]],
    embeddings: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    samples = []
    for batter_season_id, aggregate in sorted(aggregates.items()):
        embedding = embeddings.get(batter_season_id)
        row = dict(aggregate)
        row["sample_id"] = batter_season_id
        row["mechanics_signal"] = _mechanics_signal(embedding, aggregate)
        row["has_player_season_embedding"] = embedding is not None
        row["n_prior_clips"] = int((embedding or {}).get("n_clips_used") or 0)
        row["prior_mode"] = str((embedding or {}).get("prior_mode") or "none")
        row["aggregation_method"] = str((embedding or {}).get("aggregation_method") or "aggregate_bbe_mean")
        samples.append(row)
    return samples


def run_player_season_aggregate_baseline(
    base_dir: str | Path,
    *,
    prediction_run_id: str = "player_season_aggregate_mlb_2024_2026_v1",
    bbe_events: str | Path | None = None,
    player_season_embeddings: str | Path | None = None,
    player_season_batting_stats: str | Path | None = None,
    target_registry: str | Path = DEFAULT_TARGET_REGISTRY,
    player_season_embedding_feature_id: str = "player_season_embedding_v1",
    require_non_empty: bool = False,
    output_suffix: str = ".parquet",
) -> dict[str, Path]:
    """Fit a dependency-light player-season aggregate baseline."""

    base = Path(base_dir)
    bbe_path = Path(bbe_events) if bbe_events else base / "manifests/bbe_events_v1.parquet"
    batting_stats_path = (
        Path(player_season_batting_stats)
        if player_season_batting_stats
        else base / f"manifests/player_season_batting_v1{output_suffix}"
    )
    embedding_path = (
        Path(player_season_embeddings)
        if player_season_embeddings
        else base / f"features/{player_season_embedding_feature_id}/manifest{output_suffix}"
    )
    outputs = {
        "predictions": base / f"predictions/{prediction_run_id}/predictions_v1{output_suffix}",
        "metrics": base / f"predictions/{prediction_run_id}/metrics_v1.json",
        "summary": base / f"reports/preflight/player_season_aggregate_{prediction_run_id}.json",
        "label_table": base / f"datasets/player_season_targets/{prediction_run_id}/manifest{output_suffix}",
    }
    targets = load_target_registry(target_registry)
    event_rows = read_table(bbe_path) if bbe_path.exists() else []
    embedding_rows = _read_optional_table(embedding_path)
    batting_stats_rows = _read_optional_table(batting_stats_path)
    splits = _split_map(base)
    aggregates = _aggregate_bbe_targets(event_rows, splits)
    player_season_batting_matches = _merge_player_season_batting_stats(aggregates, batting_stats_rows)
    embeddings = _latest_player_season_embeddings(embedding_rows)
    samples = _build_samples(aggregates, embeddings)
    fitted = {
        target_name: _fit_linear(samples, targets[target_name].column)
        for target_name in PLAYER_SEASON_TARGETS
        if target_name in targets
    }

    predictions: list[dict[str, Any]] = []
    for sample in samples:
        for target_name in PLAYER_SEASON_TARGETS:
            if target_name not in targets:
                continue
            target = targets[target_name]
            y_true = sample.get(target.column)
            model = fitted.get(target_name)
            available_label = not _is_missing(y_true)
            available = available_label and model is not None
            y_pred = None
            missing_reason = None
            if available:
                y_pred = float(model["intercept"] + model["slope"] * float(sample["mechanics_signal"]))
                if target.kind in {"binary", "probability"}:
                    y_pred = min(max(y_pred, 0.0), 1.0)
            elif not available_label:
                missing_reason = sample.get(f"target_{target_name}_missing_reason") or "player_season_target_missing"
            else:
                missing_reason = "player_season_baseline_not_fit_for_target"
            predictions.append(
                {
                    "run_id": prediction_run_id,
                    "sample_id": str(sample["sample_id"]),
                    "event_id": None,
                    "batter_season_id": str(sample["batter_season_id"]),
                    "prediction_level": "player_season",
                    "target_name": target_name,
                    "y_true": None if _is_missing(y_true) else float(y_true),
                    "y_pred": y_pred,
                    "target_available": available,
                    "target_source": target.column,
                    "head_kind": target.kind,
                    "loss_name": target.loss,
                    "aggregation_scope": "player_season_mechanics_prior",
                    "prior_mode": str(sample["prior_mode"]),
                    "label_missing_reason": missing_reason,
                    "requires_pa_manifest": target.requires_pa_manifest,
                    "n_prior_clips": int(sample["n_prior_clips"]),
                    "aggregation_method": str(sample["aggregation_method"]),
                    "same_event_ensemble": False,
                    "prediction_std": None,
                    "split": str(sample.get("split", "unknown")),
                }
            )

    if require_non_empty and not samples:
        write_json(
            {
                "schema_version": "player_season_aggregate_summary_v1",
                "prediction_run_id": prediction_run_id,
                "error": "empty_player_season_samples",
                "bbe_events": str(bbe_path),
                "player_season_embeddings": str(embedding_path),
                "player_season_batting_stats": str(batting_stats_path),
            },
            outputs["summary"],
        )
        raise RuntimeError("player-season aggregate baseline produced 0 samples")

    validate_prediction_rows(predictions)
    metrics = evaluate_predictions(predictions, targets, run_id=prediction_run_id)
    write_table(outputs["label_table"], samples)
    write_table(outputs["predictions"], predictions)
    write_json(metrics, outputs["metrics"])
    write_json(
        {
            "schema_version": "player_season_aggregate_summary_v1",
            "prediction_run_id": prediction_run_id,
            "player_season_embedding_feature_id": player_season_embedding_feature_id,
            "bbe_events": str(bbe_path),
            "player_season_embeddings": str(embedding_path),
            "player_season_batting_stats": str(batting_stats_path),
            "event_rows": len(event_rows),
            "player_season_embedding_rows": len(embedding_rows),
            "player_season_batting_stats_rows": len(batting_stats_rows),
            "player_season_batting_matches": player_season_batting_matches,
            "player_season_samples": len(samples),
            "samples_with_player_season_embedding": sum(1 for row in samples if row["has_player_season_embedding"]),
            "prediction_rows": len(predictions),
            "fitted_targets": {key: value for key, value in fitted.items() if value is not None},
            "pa_dependent_targets": ["ba", "ops", "obp", "slg"],
            "note_ja": "BA/OPS/OBP/SLG は manifests/player_season_batting_v1 がある場合だけ年度別 player-season ラベルとして評価する。event-level head にはしない。",
            "outputs": {key: str(path) for key, path in outputs.items() if key != "summary"},
        },
        outputs["summary"],
    )
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run player-season aggregate baseline from mechanics priors.")
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/baseball_vision")
    parser.add_argument("--prediction-run-id", default="player_season_aggregate_mlb_2024_2026_v1")
    parser.add_argument("--bbe-events", default=None)
    parser.add_argument("--player-season-embeddings", default=None)
    parser.add_argument("--player-season-batting-stats", default=None)
    parser.add_argument("--target-registry", default=str(DEFAULT_TARGET_REGISTRY))
    parser.add_argument("--player-season-embedding-feature-id", default="player_season_embedding_v1")
    parser.add_argument("--require-non-empty", action="store_true")
    parser.add_argument("--output-format", choices=("parquet", "jsonl", "json", "csv"), default="parquet")
    args = parser.parse_args(argv)
    outputs = run_player_season_aggregate_baseline(
        args.base_dir,
        prediction_run_id=args.prediction_run_id,
        bbe_events=args.bbe_events,
        player_season_embeddings=args.player_season_embeddings,
        player_season_batting_stats=args.player_season_batting_stats,
        target_registry=args.target_registry,
        player_season_embedding_feature_id=args.player_season_embedding_feature_id,
        require_non_empty=args.require_non_empty,
        output_suffix="." + args.output_format,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
