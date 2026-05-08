"""Project event-level method predictions into player-season targets.

This module lets every event-level method be compared on both research axes:

1. batting-event outcomes such as EV / LA / hard-hit / barrel / xBA
2. player-season stats such as BA / OPS / OBP / SLG

The season stat predictions are intentionally lightweight and auditable. Event
predictions are first aggregated per batter-season, then PA-dependent season
stats are calibrated from those aggregate predicted mechanics/outcome features.
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
from sport_pipeline.models.player_season.aggregate_baseline import (
    _aggregate_bbe_targets,
    _merge_player_season_batting_stats,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TARGET_REGISTRY = PROJECT_ROOT / "configs/targets/target_registry_v1.yaml"
EVENT_TO_PLAYER_DIRECT = {
    "avg_ev": "pred_avg_ev",
    "avg_la": "pred_avg_la",
    "avg_xba": "pred_avg_xba",
    "avg_xwoba": "pred_avg_xwoba",
    "hard_hit_rate": "pred_hard_hit_rate",
    "barrel_rate": "pred_barrel_rate",
}
SEASON_CALIBRATION_TARGETS = ("ba", "obp", "slg", "ops")
FEATURE_COLUMNS = (
    "pred_avg_ev",
    "pred_avg_la",
    "pred_hard_hit_rate",
    "pred_barrel_rate",
    "pred_avg_xba",
    "pred_avg_xwoba",
    "log_predicted_events",
)
PLAYER_SEASON_TARGETS = (
    "avg_ev",
    "avg_la",
    "avg_xba",
    "avg_xwoba",
    "hard_hit_rate",
    "barrel_rate",
    "ba",
    "obp",
    "slg",
    "ops",
)


def projection_run_id(source_run_id: str) -> str:
    return f"{source_run_id}_player_season_projection"


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


def _prediction_path(base: Path, run_id: str) -> Path | None:
    for suffix in (".parquet", ".jsonl", ".json", ".csv"):
        path = base / "predictions" / run_id / f"predictions_v1{suffix}"
        if path.exists():
            return path
    return None


def _read_optional_table(path: Path) -> list[dict[str, Any]]:
    return read_table(path) if path.exists() else []


def _split_for_rows(rows: Iterable[dict[str, Any]]) -> str:
    counts = Counter(str(row.get("split", "unknown")) for row in rows)
    if not counts:
        return "unknown"
    return counts.most_common(1)[0][0]


def _aggregate_event_predictions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    source_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("prediction_level") != "event":
            continue
        if not row.get("target_available") or row.get("y_pred") is None:
            continue
        target_name = str(row.get("target_name"))
        if target_name not in {"ev", "la", "hard_hit", "barrel", "xba", "xwoba"}:
            continue
        batter_season_id = str(row.get("batter_season_id") or "")
        if not batter_season_id:
            continue
        pred = _to_float(row.get("y_pred"))
        if pred is None:
            continue
        values[batter_season_id][target_name].append(pred)
        source_rows[batter_season_id].append(row)

    output: dict[str, dict[str, Any]] = {}
    for batter_season_id, by_target in values.items():
        row = {
            "batter_season_id": batter_season_id,
            "predicted_event_rows": sum(len(items) for items in by_target.values()),
            "split": _split_for_rows(source_rows[batter_season_id]),
            "source_event_ids": sorted({str(item.get("event_id")) for item in source_rows[batter_season_id] if item.get("event_id") is not None}),
        }
        mapping = {
            "ev": "pred_avg_ev",
            "la": "pred_avg_la",
            "hard_hit": "pred_hard_hit_rate",
            "barrel": "pred_barrel_rate",
            "xba": "pred_avg_xba",
            "xwoba": "pred_avg_xwoba",
        }
        for event_target, column in mapping.items():
            target_values = by_target.get(event_target, [])
            row[column] = mean(target_values) if target_values else None
            row[f"n_{event_target}_predictions"] = len(target_values)
        row["log_predicted_events"] = math.log1p(max(0, int(row["predicted_event_rows"])))
        output[batter_season_id] = row
    return output


def _feature_vector(sample: dict[str, Any]) -> list[float]:
    return [1.0] + [float(sample.get(column) or 0.0) for column in FEATURE_COLUMNS]


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    n = len(vector)
    aug = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [value / scale for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0:
                continue
            aug[row] = [current - factor * pivot_value for current, pivot_value in zip(aug[row], aug[col])]
    return [aug[row][-1] for row in range(n)]


def _fit_ridge(samples: list[dict[str, Any]], target_column: str, *, l2: float = 1.0) -> dict[str, Any] | None:
    train_rows = [row for row in samples if row.get("split") == "train" and not _is_missing(row.get(target_column))]
    if len(train_rows) < 3:
        train_rows = [row for row in samples if not _is_missing(row.get(target_column))]
    if not train_rows:
        return None
    dim = len(FEATURE_COLUMNS) + 1
    xtx = [[0.0 for _ in range(dim)] for _ in range(dim)]
    xty = [0.0 for _ in range(dim)]
    for row in train_rows:
        x = _feature_vector(row)
        y = float(row[target_column])
        for i in range(dim):
            xty[i] += x[i] * y
            for j in range(dim):
                xtx[i][j] += x[i] * x[j]
    for i in range(1, dim):
        xtx[i][i] += float(l2)
    weights = _solve_linear_system(xtx, xty)
    if weights is None:
        y_mean = mean(float(row[target_column]) for row in train_rows)
        weights = [float(y_mean)] + [0.0] * (dim - 1)
    return {"weights": weights, "train_rows": len(train_rows), "feature_columns": ("intercept", *FEATURE_COLUMNS)}


def _predict_linear(model: dict[str, Any] | None, sample: dict[str, Any]) -> float | None:
    if model is None:
        return None
    weights = [float(value) for value in model["weights"]]
    x = _feature_vector(sample)
    return sum(weight * value for weight, value in zip(weights, x))


def _clamp_target(target_name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if target_name in {"avg_xba", "hard_hit_rate", "barrel_rate", "ba", "obp"}:
        return min(max(value, 0.0), 1.0)
    if target_name in {"slg", "ops"}:
        return min(max(value, 0.0), 5.0)
    return value


def _build_samples(
    predicted_aggregates: dict[str, dict[str, Any]],
    labels: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for batter_season_id, prediction in sorted(predicted_aggregates.items()):
        label = labels.get(batter_season_id)
        if label is None:
            continue
        row = {**label, **prediction}
        row["sample_id"] = batter_season_id
        samples.append(row)
    return samples


def run_event_prediction_player_season_projection(
    base_dir: str | Path,
    *,
    source_run_id: str,
    output_run_id: str | None = None,
    bbe_events: str | Path | None = None,
    source_predictions: str | Path | None = None,
    player_season_batting_stats: str | Path | None = None,
    target_registry: str | Path = DEFAULT_TARGET_REGISTRY,
    require_non_empty: bool = False,
    output_suffix: str = ".parquet",
) -> dict[str, Path]:
    """Aggregate one event-level method into player-season predictions."""

    base = Path(base_dir)
    resolved_output_run_id = output_run_id or projection_run_id(source_run_id)
    if bbe_events:
        bbe_path = Path(bbe_events)
    else:
        bbe_path = base / "manifests/bbe_events_v1.parquet"
        if not bbe_path.exists():
            bbe_path = base / "manifests/bbe_events_v1.jsonl"
    predictions_path = Path(source_predictions) if source_predictions else _prediction_path(base, source_run_id)
    batting_stats_path = (
        Path(player_season_batting_stats)
        if player_season_batting_stats
        else base / f"manifests/player_season_batting_v1{output_suffix}"
    )
    outputs = {
        "predictions": base / f"predictions/{resolved_output_run_id}/predictions_v1{output_suffix}",
        "metrics": base / f"predictions/{resolved_output_run_id}/metrics_v1.json",
        "summary": base / f"reports/preflight/player_season_projection_{resolved_output_run_id}.json",
        "label_table": base / f"datasets/player_season_targets/{resolved_output_run_id}/manifest{output_suffix}",
    }
    if predictions_path is None or not predictions_path.exists():
        write_json(
            {
                "schema_version": "event_prediction_player_season_projection_summary_v1",
                "source_run_id": source_run_id,
                "output_run_id": resolved_output_run_id,
                "status": "missing_source_predictions",
            },
            outputs["summary"],
        )
        if require_non_empty:
            raise FileNotFoundError(f"source predictions not found for {source_run_id}")
        return outputs

    targets = load_target_registry(target_registry)
    event_rows = read_table(bbe_path) if bbe_path.exists() else []
    source_rows = read_table(predictions_path)
    validate_prediction_rows(source_rows)
    batting_stats_rows = _read_optional_table(batting_stats_path)

    labels = _aggregate_bbe_targets(event_rows, {})
    matches = _merge_player_season_batting_stats(labels, batting_stats_rows)
    predicted_aggregates = _aggregate_event_predictions(source_rows)
    samples = _build_samples(predicted_aggregates, labels)
    fitted = {
        target_name: _fit_ridge(samples, targets[target_name].column)
        for target_name in SEASON_CALIBRATION_TARGETS
        if target_name in targets
    }

    predictions: list[dict[str, Any]] = []
    for sample in samples:
        for target_name in PLAYER_SEASON_TARGETS:
            if target_name not in targets:
                continue
            target = targets[target_name]
            y_true = sample.get(target.column)
            direct_column = EVENT_TO_PLAYER_DIRECT.get(target_name)
            if direct_column is not None:
                y_pred = _to_float(sample.get(direct_column))
                aggregation_method = "mean_event_predictions"
            elif target_name == "ba" and sample.get("pred_avg_xba") is not None and fitted.get(target_name) is None:
                y_pred = _to_float(sample.get("pred_avg_xba"))
                aggregation_method = "avg_xba_proxy"
            else:
                y_pred = _predict_linear(fitted.get(target_name), sample)
                aggregation_method = "aggregate_event_predictions_then_ridge_calibration"
            y_pred = _clamp_target(target_name, y_pred)
            available_label = not _is_missing(y_true)
            available = available_label and y_pred is not None
            missing_reason = None
            if not available_label:
                missing_reason = sample.get(f"target_{target_name}_missing_reason") or "player_season_target_missing"
            elif y_pred is None:
                missing_reason = "event_prediction_projection_not_fit_for_target"
            predictions.append(
                {
                    "run_id": resolved_output_run_id,
                    "sample_id": f"{sample['sample_id']}__{target_name}__{source_run_id}",
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
                    "aggregation_scope": "player_season_from_event_predictions",
                    "prior_mode": "event_predictions_aggregated_by_batter_season",
                    "label_missing_reason": missing_reason,
                    "requires_pa_manifest": target.requires_pa_manifest,
                    "n_prior_clips": 0,
                    "aggregation_method": aggregation_method,
                    "same_event_ensemble": False,
                    "prediction_std": None,
                    "split": str(sample.get("split", "unknown")),
                }
            )

    if require_non_empty and not samples:
        write_json(
            {
                "schema_version": "event_prediction_player_season_projection_summary_v1",
                "source_run_id": source_run_id,
                "output_run_id": resolved_output_run_id,
                "status": "empty_samples",
                "source_predictions": str(predictions_path),
                "event_rows": len(event_rows),
                "source_rows": len(source_rows),
                "predicted_player_seasons": len(predicted_aggregates),
                "label_player_seasons": len(labels),
            },
            outputs["summary"],
        )
        raise RuntimeError(f"player-season projection for {source_run_id} produced 0 samples")

    validate_prediction_rows(predictions)
    metrics = evaluate_predictions(predictions, targets, run_id=resolved_output_run_id)
    write_table(outputs["label_table"], samples)
    write_table(outputs["predictions"], predictions)
    write_json(metrics, outputs["metrics"])
    write_json(
        {
            "schema_version": "event_prediction_player_season_projection_summary_v1",
            "source_run_id": source_run_id,
            "output_run_id": resolved_output_run_id,
            "status": "complete",
            "source_predictions": str(predictions_path),
            "bbe_events": str(bbe_path),
            "player_season_batting_stats": str(batting_stats_path),
            "source_prediction_rows": len(source_rows),
            "source_event_prediction_rows": sum(1 for row in source_rows if row.get("prediction_level") == "event"),
            "predicted_player_seasons": len(predicted_aggregates),
            "label_player_seasons": len(labels),
            "player_season_batting_matches": matches,
            "player_season_samples": len(samples),
            "prediction_rows": len(predictions),
            "fitted_targets": {key: value for key, value in fitted.items() if value is not None},
            "direct_targets": EVENT_TO_PLAYER_DIRECT,
            "calibrated_targets": list(SEASON_CALIBRATION_TARGETS),
            "note_ja": "event-level 予測を batter-season ごとに集約し、BA/OBP/SLG/OPS は年度別 player-season batting stats に対して軽量 calibration で評価する。",
            "outputs": {key: str(path) for key, path in outputs.items() if key != "summary"},
        },
        outputs["summary"],
    )
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Project event-level predictions into player-season stat predictions.")
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/baseball_vision")
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--output-run-id", default=None)
    parser.add_argument("--bbe-events", default=None)
    parser.add_argument("--source-predictions", default=None)
    parser.add_argument("--player-season-batting-stats", default=None)
    parser.add_argument("--target-registry", default=str(DEFAULT_TARGET_REGISTRY))
    parser.add_argument("--require-non-empty", action="store_true")
    parser.add_argument("--output-format", choices=("parquet", "jsonl", "json", "csv"), default="parquet")
    args = parser.parse_args(argv)
    outputs = run_event_prediction_player_season_projection(
        args.base_dir,
        source_run_id=args.source_run_id,
        output_run_id=args.output_run_id,
        bbe_events=args.bbe_events,
        source_predictions=args.source_predictions,
        player_season_batting_stats=args.player_season_batting_stats,
        target_registry=args.target_registry,
        require_non_empty=args.require_non_empty,
        output_suffix="." + args.output_format,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
