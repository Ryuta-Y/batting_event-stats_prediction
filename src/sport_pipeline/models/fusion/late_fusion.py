"""Dependency-free late fusion for predictions_v1 rows."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

from sport_pipeline.evaluation.predictions import validate_prediction_rows
from sport_pipeline.models.fusion.contracts import FUSION_CONTRACT_VERSION


DEFAULT_SCOPE_WEIGHTS = {
    "context_only": 0.75,
    "current_event_only": 1.0,
    "current_event_with_player_season_prior": 1.1,
    "raw_video_lightweight": 0.60,
    "video_frozen_encoder": 0.85,
    "image_frozen_encoder": 0.75,
    "raw_video_finetune": 0.90,
    "player_season_mechanics_prior": 0.80,
    "vlm_mechanics_features": 0.70,
    "audio_raw_impact": 0.55,
    "audio_enhanced_impact": 0.60,
    "audio_separated_impact": 0.60,
    "audio_embedding_impact": 0.70,
    "same_event_view_crop_augmentation_ensemble": 0.95,
}


@dataclass(frozen=True)
class FusionResult:
    prediction_rows: list[dict]
    audit_rows: list[dict]


def _fusion_key(row: dict) -> tuple[str, str | None, str, str]:
    level = str(row["prediction_level"])
    target = str(row["target_name"])
    batter_season_id = str(row["batter_season_id"])
    if target == "ops" and level == "event":
        raise ValueError("OPS must not be fused as an event-level target")
    if level == "event":
        event_id = row.get("event_id")
        if event_id is None:
            raise ValueError("event-level fusion rows require event_id")
        return (level, str(event_id), batter_season_id, target)
    if level == "player_season":
        return (level, None, batter_season_id, target)
    raise ValueError(f"unsupported prediction_level for fusion: {level}")


def _ordered_unique(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def _weight_for_row(row: dict, scope_weights: dict[str, float]) -> float:
    scope = str(row.get("aggregation_scope", "unknown"))
    target_scope = f"{row.get('target_name')}::{scope}"
    base = float(scope_weights.get(target_scope, scope_weights.get(scope, 0.5)))
    if row.get("target_available") is False or row.get("y_pred") is None:
        return 0.0
    prediction_std = row.get("prediction_std")
    if prediction_std is None:
        return base
    uncertainty = max(float(prediction_std), 1e-6)
    return base / (1.0 + uncertainty)


def learn_validation_scope_weights(rows: Iterable[dict], min_rows: int = 2) -> dict[str, float]:
    """Learn per-target/scope late-fusion weights from validation errors.

    The output keys are either `target::aggregation_scope` or aggregation scope.
    Missing labels and unavailable predictions are ignored, so xBA/xwOBA masks
    remain intact and OPS is learned only if valid player-season rows exist.
    """

    error_by_target_scope: dict[tuple[str, str], list[float]] = {}
    error_by_scope: dict[str, list[float]] = {}
    for row in rows:
        split = str(row.get("split", "")).lower()
        if split not in {"validation", "val", "dev", "holdout"}:
            continue
        if not row.get("target_available") or row.get("y_true") is None or row.get("y_pred") is None:
            continue
        target = str(row["target_name"])
        scope = str(row.get("aggregation_scope", "unknown"))
        y_true = float(row["y_true"])
        y_pred = float(row["y_pred"])
        if str(row.get("head_kind")) in {"binary", "probability"}:
            error = (y_pred - y_true) ** 2
        else:
            error = abs(y_pred - y_true)
        error_by_target_scope.setdefault((target, scope), []).append(error)
        error_by_scope.setdefault(scope, []).append(error)
    weights: dict[str, float] = {}
    for (target, scope), errors in error_by_target_scope.items():
        if len(errors) >= min_rows:
            weights[f"{target}::{scope}"] = 1.0 / (sum(errors) / len(errors) + 1e-6)
    for scope, errors in error_by_scope.items():
        if len(errors) >= min_rows:
            weights[scope] = 1.0 / (sum(errors) / len(errors) + 1e-6)
    return weights


def _resolve_prior_mode(rows: list[dict]) -> str:
    modes = _ordered_unique(str(row.get("prior_mode", "none")) for row in rows)
    if len(modes) == 1:
        return modes[0]
    return "mixed:" + "+".join(modes)


def fuse_prediction_group(
    rows: list[dict],
    fusion_run_id: str,
    scope_weights: dict[str, float] | None = None,
    aggregation_method: str = "late_fusion_weighted_average",
) -> FusionResult:
    """Fuse rows that share one prediction level, target, and event/player-season key."""

    if not rows:
        raise ValueError("cannot fuse empty row group")
    keys = {_fusion_key(row) for row in rows}
    if len(keys) != 1:
        raise ValueError("fusion group must not mix events, player-seasons, levels, or targets")
    scope_weights = scope_weights or DEFAULT_SCOPE_WEIGHTS
    level, event_id, batter_season_id, target_name = next(iter(keys))
    validate_prediction_rows(rows)

    available_rows = [row for row in rows if row.get("target_available") and row.get("y_pred") is not None]
    sample_id = (
        f"{event_id}__{target_name}__late_fusion"
        if level == "event"
        else f"{batter_season_id}__{target_name}__late_fusion"
    )
    base_row = available_rows[0] if available_rows else rows[0]
    audit_rows = [
        {
            "schema_version": FUSION_CONTRACT_VERSION,
            "fusion_run_id": fusion_run_id,
            "fusion_sample_id": sample_id,
            "source_run_id": str(row["run_id"]),
            "source_sample_id": str(row["sample_id"]),
            "source_event_id": row.get("event_id"),
            "source_batter_season_id": str(row["batter_season_id"]),
            "source_prediction_level": str(row["prediction_level"]),
            "source_target_name": str(row["target_name"]),
            "source_aggregation_scope": str(row.get("aggregation_scope", "unknown")),
            "source_prior_mode": str(row.get("prior_mode", "none")),
            "source_same_event_ensemble": bool(row.get("same_event_ensemble", False)),
            "source_n_prior_clips": int(row.get("n_prior_clips") or 0),
            "source_prediction_std": row.get("prediction_std"),
            "source_target_available": bool(row.get("target_available", False)),
            "source_label_missing_reason": row.get("label_missing_reason"),
            "fusion_weight": _weight_for_row(row, scope_weights),
        }
        for row in rows
    ]

    if not available_rows:
        reasons = _ordered_unique(
            str(row.get("label_missing_reason") or "fusion_input_unavailable") for row in rows
        )
        prediction = {
            "run_id": fusion_run_id,
            "sample_id": sample_id,
            "event_id": event_id,
            "batter_season_id": batter_season_id,
            "prediction_level": level,
            "target_name": target_name,
            "y_true": None,
            "y_pred": None,
            "target_available": False,
            "target_source": str(base_row.get("target_source", "fusion")),
            "head_kind": str(base_row.get("head_kind", "regression")),
            "loss_name": str(base_row.get("loss_name", "none")),
            "aggregation_scope": f"late_fusion_{level}",
            "prior_mode": _resolve_prior_mode(rows),
            "label_missing_reason": "+".join(reasons),
            "requires_pa_manifest": bool(base_row.get("requires_pa_manifest", False)),
            "n_prior_clips": max(int(row.get("n_prior_clips") or 0) for row in rows),
            "aggregation_method": aggregation_method,
            "same_event_ensemble": any(bool(row.get("same_event_ensemble", False)) for row in rows),
            "prediction_std": None,
        }
        return FusionResult([prediction], audit_rows)

    weights = [_weight_for_row(row, scope_weights) for row in available_rows]
    if sum(weights) <= 0:
        weights = [1.0 for _ in available_rows]
    denominator = sum(weights)
    values = [float(row["y_pred"]) for row in available_rows]
    fused_pred = sum(value * weight for value, weight in zip(values, weights)) / denominator
    weighted_variance = sum(weight * (value - fused_pred) ** 2 for value, weight in zip(values, weights)) / denominator
    input_variance = sum(
        weight * ((float(row["prediction_std"]) ** 2) if row.get("prediction_std") is not None else 0.0)
        for row, weight in zip(available_rows, weights)
    ) / denominator
    y_true = next((row.get("y_true") for row in available_rows if row.get("y_true") is not None), None)
    prediction = {
        "run_id": fusion_run_id,
        "sample_id": sample_id,
        "event_id": event_id,
        "batter_season_id": batter_season_id,
        "prediction_level": level,
        "target_name": target_name,
        "y_true": y_true,
        "y_pred": fused_pred,
        "target_available": True,
        "target_source": str(base_row.get("target_source", "fusion")),
        "head_kind": str(base_row.get("head_kind", "regression")),
        "loss_name": str(base_row.get("loss_name", "huber")),
        "aggregation_scope": f"late_fusion_{level}",
        "prior_mode": _resolve_prior_mode(rows),
        "label_missing_reason": None,
        "requires_pa_manifest": bool(base_row.get("requires_pa_manifest", False)),
        "n_prior_clips": max(int(row.get("n_prior_clips") or 0) for row in rows),
        "aggregation_method": aggregation_method,
        "same_event_ensemble": any(bool(row.get("same_event_ensemble", False)) for row in rows),
        "prediction_std": sqrt(weighted_variance + input_variance),
    }
    return FusionResult([prediction], audit_rows)


def late_fuse_prediction_rows(
    rows: Iterable[dict],
    fusion_run_id: str,
    scope_weights: dict[str, float] | None = None,
    aggregation_method: str = "late_fusion_weighted_average",
) -> FusionResult:
    """Fuse predictions_v1 rows per target without crossing event boundaries."""

    row_list = list(rows)
    validate_prediction_rows(row_list)
    grouped: dict[tuple[str, str | None, str, str], list[dict]] = {}
    for row in row_list:
        grouped.setdefault(_fusion_key(row), []).append(row)

    predictions: list[dict] = []
    audit: list[dict] = []
    for group_rows in grouped.values():
        result = fuse_prediction_group(
            group_rows,
            fusion_run_id=fusion_run_id,
            scope_weights=scope_weights,
            aggregation_method=aggregation_method,
        )
        predictions.extend(result.prediction_rows)
        audit.extend(result.audit_rows)
    validate_prediction_rows(predictions)
    return FusionResult(predictions, audit)
