"""Player-season prior dataset helpers."""

from __future__ import annotations

from typing import Any, Iterable

from sport_pipeline.datasets.sequence.contracts import SEQUENCE_DATASET_CONTRACT_VERSION
from sport_pipeline.features.embeddings import ensure_numeric_vector
from sport_pipeline.models.sequence.aggregation import aggregate_clip_embeddings


PRIOR_MODES = ("none", "same_season_train_only", "past_only", "oracle_full_season")
EVENT_WITH_PRIOR_SCOPE = "current_event_with_player_season_prior"
EVENT_ONLY_SCOPE = "current_event_only"


def _is_clean_prior_candidate(row: dict[str, Any]) -> bool:
    return (
        row.get("clip_status") == "clean_clip"
        and row.get("quality_tier") == "usable_primary"
        and bool(row.get("clean_cohort_eligible"))
        and row.get("target_alignment_status") == "event_aligned"
    )


def select_prior_clip_embeddings(
    current_event: dict[str, Any],
    clip_embeddings: Iterable[dict[str, Any]],
    prior_mode: str,
    exclude_current_event: bool = True,
) -> list[dict[str, Any]]:
    """Select prior clips without mixing same-event ensemble into player prior."""

    if prior_mode not in PRIOR_MODES:
        raise ValueError(f"unknown prior_mode: {prior_mode}")
    if prior_mode == "none":
        return []

    selected: list[dict[str, Any]] = []
    current_batter_season_id = current_event["batter_season_id"]
    current_event_id = current_event["event_id"]
    current_game_date = current_event.get("game_date")
    for row in clip_embeddings:
        if row.get("batter_season_id") != current_batter_season_id:
            continue
        if exclude_current_event and row.get("event_id") == current_event_id:
            continue
        if not _is_clean_prior_candidate(row):
            continue
        if prior_mode == "same_season_train_only" and row.get("split") != "train":
            continue
        if prior_mode == "past_only" and not (row.get("game_date") < current_game_date):
            continue
        selected.append(row)
    return selected


def build_event_with_prior_row(
    current_event: dict[str, Any],
    current_clip_embedding: dict[str, Any],
    all_clip_embeddings: Iterable[dict[str, Any]],
    prior_mode: str,
    aggregation_method: str = "quality_weighted_pooling",
    target_registry_version: str = "target_registry_v1",
) -> dict[str, Any]:
    """Build one event_with_player_prior_v1 row for event-level prediction."""

    selected = select_prior_clip_embeddings(current_event, all_clip_embeddings, prior_mode)
    if selected:
        prior = aggregate_clip_embeddings(
            selected,
            aggregation_method=aggregation_method,
            current_event=current_event,
            prior_mode=prior_mode,
        )
        prior_values = prior["embedding_values"]
        prior_path = prior["embedding_path"]
        uses_future = prior["uses_future_clips"]
    else:
        prior_values = []
        prior_path = None
        uses_future = False

    current_values = ensure_numeric_vector(current_clip_embedding["embedding_values"])
    sample_id = f"{current_event['event_id']}__{prior_mode}__{aggregation_method}"
    return {
        "schema_version": SEQUENCE_DATASET_CONTRACT_VERSION,
        "sample_id": sample_id,
        "event_id": current_event["event_id"],
        "clip_id": current_clip_embedding["clip_id"],
        "sequence_id": current_clip_embedding["sequence_id"],
        "same_event_group_id": current_event["same_event_group_id"],
        "batter_id": current_event["batter_id"],
        "season": current_event["season"],
        "batter_season_id": current_event["batter_season_id"],
        "game_date": current_event["game_date"],
        "split": current_event.get("split", current_clip_embedding.get("split", "unknown")),
        "context_feature_path": current_event.get("context_feature_path"),
        "current_clip_embedding_path": current_clip_embedding.get("embedding_path"),
        "current_clip_embedding_values": current_values,
        "player_season_embedding_path": prior_path,
        "player_season_embedding_values": prior_values,
        "prior_mode": prior_mode,
        "aggregation_method": aggregation_method,
        "aggregation_scope": EVENT_WITH_PRIOR_SCOPE if prior_mode != "none" else EVENT_ONLY_SCOPE,
        "n_prior_clips": len(selected),
        "prior_clip_ids": [row["clip_id"] for row in selected],
        "uses_future_clips": uses_future,
        "prediction_level": "event",
        "same_event_ensemble": False,
        "target_registry_version": target_registry_version,
        "target_ev_available": bool(current_event.get("target_ev_available", True)),
        "target_la_available": bool(current_event.get("target_la_available", True)),
        "target_hard_hit_available": bool(current_event.get("target_hard_hit_available", True)),
        "target_barrel_available": bool(current_event.get("target_barrel_available", True)),
        "target_xba_available": bool(current_event.get("target_xba_available", False)),
        "target_xwoba_available": bool(current_event.get("target_xwoba_available", False)),
        "target_ops_available": False,
        "label_missing_reason": current_event.get("label_missing_reason"),
    }

