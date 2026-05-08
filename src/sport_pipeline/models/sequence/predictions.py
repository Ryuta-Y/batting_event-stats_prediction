"""Mapping sequence model outputs into predictions_v1 rows."""

from __future__ import annotations

from typing import Any


EVENT_TARGETS = {"ev", "la", "hard_hit", "barrel", "xba", "xwoba"}


def build_sequence_prediction_row(
    run_id: str,
    event_prior_row: dict[str, Any],
    target_name: str,
    y_pred: float | None,
    y_true: float | None = None,
    target_available: bool = True,
    target_source: str = "statcast_column",
    head_kind: str = "regression",
    loss_name: str = "huber",
    label_missing_reason: str | None = None,
    requires_pa_manifest: bool = False,
    prediction_std: float | None = None,
) -> dict[str, Any]:
    """Build a D1 predictions_v1-compatible row for sequence/event-prior outputs."""

    if target_name == "ops":
        raise ValueError("OPS must not be emitted as an event-level sequence head")
    if target_name not in EVENT_TARGETS:
        raise ValueError(f"unknown event-level target for sequence model: {target_name}")
    return {
        "run_id": run_id,
        "sample_id": event_prior_row["sample_id"],
        "event_id": event_prior_row["event_id"],
        "batter_season_id": event_prior_row["batter_season_id"],
        "prediction_level": "event",
        "target_name": target_name,
        "y_true": y_true,
        "y_pred": y_pred,
        "target_available": target_available,
        "target_source": target_source,
        "head_kind": head_kind,
        "loss_name": loss_name,
        "aggregation_scope": event_prior_row["aggregation_scope"],
        "prior_mode": event_prior_row["prior_mode"],
        "label_missing_reason": label_missing_reason,
        "requires_pa_manifest": requires_pa_manifest,
        "n_prior_clips": event_prior_row["n_prior_clips"],
        "aggregation_method": event_prior_row["aggregation_method"],
        "same_event_ensemble": event_prior_row["same_event_ensemble"],
        "prediction_std": prediction_std,
    }

