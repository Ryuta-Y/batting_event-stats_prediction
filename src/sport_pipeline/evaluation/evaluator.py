"""Common evaluator for predictions_v1 rows."""

from __future__ import annotations

from collections import defaultdict
from math import sqrt
from statistics import mean
from typing import Any

from sport_pipeline.evaluation.predictions import validate_prediction_rows
from sport_pipeline.evaluation.target_registry import TargetSpec


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x = mean(xs)
    mean_y = mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = sum((x - mean_x) ** 2 for x in xs)
    denom_y = sum((y - mean_y) ** 2 for y in ys)
    if denom_x == 0 or denom_y == 0:
        return None
    return numerator / sqrt(denom_x * denom_y)


def _regression_metrics(y_true: list[float], y_pred: list[float], requested: tuple[str, ...]) -> dict[str, float | None]:
    errors = [pred - truth for truth, pred in zip(y_true, y_pred)]
    metrics: dict[str, float | None] = {
        "mae": mean(abs(error) for error in errors),
        "rmse": sqrt(mean(error * error for error in errors)),
    }
    if "r2" in requested:
        y_mean = mean(y_true)
        ss_res = sum(error * error for error in errors)
        ss_tot = sum((truth - y_mean) ** 2 for truth in y_true)
        metrics["r2"] = None if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
    if "spearman" in requested:
        metrics["spearman"] = _pearson(_rank(y_true), _rank(y_pred))
    return metrics


def _binary_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float | None]:
    labels = [1 if value >= 0.5 else 0 for value in y_true]
    preds = [1 if value >= 0.5 else 0 for value in y_pred]
    tp = sum(1 for label, pred in zip(labels, preds) if label == 1 and pred == 1)
    fp = sum(1 for label, pred in zip(labels, preds) if label == 0 and pred == 1)
    fn = sum(1 for label, pred in zip(labels, preds) if label == 1 and pred == 0)
    precision_den = tp + fp
    recall_den = tp + fn
    precision = None if precision_den == 0 else tp / precision_den
    recall = None if recall_den == 0 else tp / recall_den
    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    brier = mean((pred - label) ** 2 for label, pred in zip(labels, y_pred))
    return {"f1": f1, "brier": brier}


def _skip_reason(row: dict[str, Any], target: TargetSpec | None) -> str | None:
    if target is None:
        return "unknown_target"
    if row.get("prediction_level") != target.level:
        return "prediction_level_mismatch"
    if target.requires_pa_manifest and row.get("requires_pa_manifest") is True and not row.get("target_available"):
        return row.get("label_missing_reason") or "pa_manifest_unavailable"
    if not row.get("target_available"):
        return row.get("label_missing_reason") or "target_unavailable"
    if row.get("y_true") is None:
        return row.get("label_missing_reason") or "missing_y_true"
    if row.get("y_pred") is None:
        return "missing_y_pred"
    return None


def evaluate_predictions(
    rows: list[dict[str, Any]],
    targets: dict[str, TargetSpec],
    run_id: str,
) -> dict[str, Any]:
    """Compute metrics from predictions_v1 rows, skipping unavailable labels."""

    validate_prediction_rows(rows)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    availability: dict[str, dict[str, int]] = defaultdict(lambda: {"available": 0, "missing": 0})
    skipped: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in rows:
        target_name = row["target_name"]
        target = targets.get(target_name)
        reason = _skip_reason(row, target)
        if reason is not None:
            availability[target_name]["missing"] += 1
            skipped[target_name][reason] += 1
            continue
        availability[target_name]["available"] += 1
        grouped[(row["prediction_level"], target_name)].append(row)

    metrics: dict[str, dict[str, Any]] = defaultdict(dict)
    for (level, target_name), target_rows in grouped.items():
        target = targets[target_name]
        y_true = [float(row["y_true"]) for row in target_rows]
        y_pred = [float(row["y_pred"]) for row in target_rows]
        if target.kind == "binary":
            values = _binary_metrics(y_true, y_pred)
        else:
            values = _regression_metrics(y_true, y_pred, target.metrics)
        values["n_available"] = len(target_rows)
        values["n_skipped"] = sum(skipped[target_name].values())
        metrics[level][target_name] = values

    return {
        "schema_version": "metrics_v1",
        "run_id": run_id,
        "metrics": dict(metrics),
        "label_availability": dict(availability),
        "skipped": {target: dict(reasons) for target, reasons in skipped.items()},
    }

