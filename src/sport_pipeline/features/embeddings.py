"""Clip embedding metadata helpers."""

from __future__ import annotations

from typing import Any

from sport_pipeline.features.sequence_contracts import SEQUENCE_CONTRACT_VERSION


QUALITY_TIER_WEIGHTS = {
    "usable_primary": 1.0,
    "usable_with_outliers": 0.55,
    "review_only": 0.15,
    "excluded": 0.0,
}


def ensure_numeric_vector(values: list[float] | tuple[float, ...]) -> list[float]:
    """Validate and normalize a numeric embedding vector."""

    if not values:
        raise ValueError("embedding vector must not be empty")
    vector: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("embedding values must be numeric")
        vector.append(float(value))
    return vector


def clip_quality_weight(row: dict[str, Any]) -> float:
    """Compute a deterministic quality weight for pooling baselines."""

    tier_weight = QUALITY_TIER_WEIGHTS.get(str(row.get("quality_tier", "review_only")), 0.0)
    if tier_weight == 0.0:
        return 0.0
    contact = float(row.get("contact_confidence", 0.0))
    view = float(row.get("view_confidence", 0.0))
    pose = float(row.get("pose_coverage", 0.0))
    clean_bonus = 1.0 if row.get("clean_cohort_eligible") else 0.75
    return tier_weight * max(0.0, contact) * max(0.0, view) * max(0.0, pose) * clean_bonus


def build_clip_embedding_row(
    sequence_row: dict[str, Any],
    embedding_values: list[float],
    encoder_name: str = "tcn_sequence_encoder",
    encoder_version: str = "interface_v1",
    embedding_path: str | None = None,
    contact_confidence: float = 1.0,
    view_confidence: float = 1.0,
    pose_coverage: float = 1.0,
    clean_cohort_eligible: bool = True,
    target_alignment_status: str = "event_aligned",
) -> dict[str, Any]:
    """Build one clip_embedding_v1 manifest row from a sequence row."""

    vector = ensure_numeric_vector(embedding_values)
    return {
        "schema_version": SEQUENCE_CONTRACT_VERSION,
        "clip_id": sequence_row["clip_id"],
        "sequence_id": sequence_row["sequence_id"],
        "event_id": sequence_row["event_id"],
        "same_event_group_id": sequence_row["same_event_group_id"],
        "view_id": sequence_row["view_id"],
        "batter_id": sequence_row["batter_id"],
        "season": sequence_row["season"],
        "batter_season_id": sequence_row["batter_season_id"],
        "game_date": sequence_row["game_date"],
        "split": sequence_row["split"],
        "encoder_name": encoder_name,
        "encoder_version": encoder_version,
        "embedding_path": embedding_path,
        "embedding_values": vector,
        "embedding_dim": len(vector),
        "quality_tier": sequence_row["quality_tier"],
        "clip_status": sequence_row["clip_status"],
        "view_label": sequence_row["view_label"],
        "view_confidence": view_confidence,
        "contact_confidence": contact_confidence,
        "pose_coverage": pose_coverage,
        "clean_cohort_eligible": clean_cohort_eligible,
        "target_alignment_status": target_alignment_status,
    }

