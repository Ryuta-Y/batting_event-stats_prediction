"""Pooling utilities for same-event ensembles and player-season priors."""

from __future__ import annotations

import math
from typing import Any, Iterable

from sport_pipeline.features.embeddings import clip_quality_weight, ensure_numeric_vector
from sport_pipeline.features.sequence_contracts import SEQUENCE_CONTRACT_VERSION


AGGREGATION_METHODS = (
    "mean_pooling",
    "quality_weighted_pooling",
    "attention_pooling",
    "learned_attention_pooling",
    "set_transformer_pooling",
)


def _check_same_dim(vectors: list[list[float]]) -> int:
    if not vectors:
        raise ValueError("at least one vector is required")
    dim = len(vectors[0])
    if any(len(vector) != dim for vector in vectors):
        raise ValueError("all vectors must have the same dimension")
    return dim


def mean_pool(vectors: Iterable[list[float] | tuple[float, ...]]) -> list[float]:
    """Permutation-invariant mean pooling baseline."""

    normalized = [ensure_numeric_vector(vector) for vector in vectors]
    dim = _check_same_dim(normalized)
    return [sum(vector[i] for vector in normalized) / len(normalized) for i in range(dim)]


def quality_weighted_pool(
    vectors: Iterable[list[float] | tuple[float, ...]],
    weights: Iterable[float],
) -> list[float]:
    """Quality-weighted pooling with deterministic fallback to mean."""

    normalized = [ensure_numeric_vector(vector) for vector in vectors]
    normalized_weights = [max(0.0, float(weight)) for weight in weights]
    dim = _check_same_dim(normalized)
    if len(normalized_weights) != len(normalized):
        raise ValueError("weights length must match vectors length")
    denominator = sum(normalized_weights)
    if denominator <= 0:
        return mean_pool(normalized)
    return [
        sum(vector[i] * weight for vector, weight in zip(normalized, normalized_weights)) / denominator
        for i in range(dim)
    ]


def attention_pool(
    vectors: Iterable[list[float] | tuple[float, ...]],
    quality_weights: Iterable[float] | None = None,
) -> list[float]:
    """Small deterministic attention-style pooling placeholder.

    This is not a learned model. It provides the same interface as a future
    attention pooling module while keeping local tests dependency-free.
    """

    normalized = [ensure_numeric_vector(vector) for vector in vectors]
    _check_same_dim(normalized)
    if quality_weights is None:
        quality_values = [1.0] * len(normalized)
    else:
        quality_values = [max(0.0, float(weight)) for weight in quality_weights]
        if len(quality_values) != len(normalized):
            raise ValueError("quality_weights length must match vectors length")
    scores = [sum(vector) / len(vector) + math.log1p(weight) for vector, weight in zip(normalized, quality_values)]
    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in scores]
    denominator = sum(exp_scores)
    return quality_weighted_pool(normalized, [score / denominator for score in exp_scores])


def aggregate_clip_embeddings(
    clip_embedding_rows: Iterable[dict[str, Any]],
    aggregation_method: str,
    current_event: dict[str, Any],
    prior_mode: str,
    quality_policy: str = "usable_primary_only",
) -> dict[str, Any]:
    """Aggregate clean clip embeddings into one player-season prior row."""

    rows = list(clip_embedding_rows)
    if aggregation_method not in AGGREGATION_METHODS:
        raise ValueError(f"unknown aggregation_method: {aggregation_method}")
    if not rows:
        raise ValueError("cannot aggregate empty clip set")
    vectors = [ensure_numeric_vector(row["embedding_values"]) for row in rows]
    weights = [clip_quality_weight(row) for row in rows]
    if aggregation_method == "mean_pooling":
        embedding = mean_pool(vectors)
    elif aggregation_method == "quality_weighted_pooling":
        embedding = quality_weighted_pool(vectors, weights)
    elif aggregation_method == "attention_pooling":
        embedding = attention_pool(vectors, weights)
    else:
        # Learned prior modules are trained in Colab and can read/write the same
        # artifact contract. For dependency-free artifact building, keep a stable
        # attention-weighted fallback and preserve the requested method in metadata.
        embedding = attention_pool(vectors, weights)

    current_date = current_event.get("game_date")
    uses_future_clips = any(row.get("game_date") > current_date for row in rows) if current_date else False
    batter_season_id = str(current_event["batter_season_id"])
    cutoff_date = current_date if prior_mode == "past_only" else current_event.get("cutoff_date", current_date)
    return {
        "schema_version": SEQUENCE_CONTRACT_VERSION,
        "batter_season_id": batter_season_id,
        "batter_id": current_event["batter_id"],
        "season": current_event["season"],
        "aggregator_name": "player_season_set_aggregator",
        "aggregator_version": "v1",
        "aggregation_method": aggregation_method,
        "aggregation_scope": "player_season_mechanics_prior",
        "prior_mode": prior_mode,
        "cutoff_event_id": current_event.get("event_id"),
        "cutoff_date": cutoff_date,
        "n_clips_total": len(rows),
        "n_clips_used": len(rows),
        "clip_ids_used": [row["clip_id"] for row in rows],
        "source_event_ids_used": [row["event_id"] for row in rows],
        "uses_future_clips": uses_future_clips,
        "embedding_path": None,
        "embedding_values": embedding,
        "embedding_dim": len(embedding),
        "quality_policy": quality_policy,
        "split_policy": prior_mode,
    }


def same_event_ensemble_predictions(
    prediction_rows: Iterable[dict[str, Any]],
    target_name: str,
    aggregation_method: str = "quality_weighted_pooling",
) -> dict[str, Any]:
    """Average predictions only when all rows refer to the same event target."""

    rows = [row for row in prediction_rows if row.get("target_name") == target_name]
    if not rows:
        raise ValueError("no rows for target")
    event_ids = {row.get("event_id") for row in rows}
    if len(event_ids) != 1:
        raise ValueError("same-event ensemble cannot average different events")
    same_event_groups = {row.get("same_event_group_id", row.get("event_id")) for row in rows}
    if len(same_event_groups) != 1:
        raise ValueError("same-event ensemble requires one same_event_group_id")
    y_preds = [float(row["y_pred"]) for row in rows]
    weights = [float(row.get("ensemble_weight", 1.0)) for row in rows]
    y_pred = quality_weighted_pool([[value] for value in y_preds], weights)[0]
    variance = sum((value - y_pred) ** 2 for value in y_preds) / len(y_preds)
    first = rows[0]
    return {
        "run_id": first["run_id"],
        "sample_id": f"{first['event_id']}__same_event_ensemble__{target_name}",
        "event_id": first["event_id"],
        "batter_season_id": first["batter_season_id"],
        "prediction_level": "event",
        "target_name": target_name,
        "y_true": first.get("y_true"),
        "y_pred": y_pred,
        "target_available": bool(first.get("target_available", True)),
        "target_source": first["target_source"],
        "head_kind": first["head_kind"],
        "loss_name": first["loss_name"],
        "aggregation_scope": "same_event_view_crop_augmentation_ensemble",
        "prior_mode": "none",
        "label_missing_reason": first.get("label_missing_reason"),
        "requires_pa_manifest": first.get("requires_pa_manifest", False),
        "n_prior_clips": 0,
        "aggregation_method": aggregation_method,
        "same_event_ensemble": True,
        "prediction_std": variance ** 0.5,
    }
