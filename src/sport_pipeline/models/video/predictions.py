"""Map frozen image/video baseline outputs to predictions_v1."""

from __future__ import annotations

from typing import Mapping

from sport_pipeline.models.video.heads import BaselineHeadSpec


def build_visual_prediction_rows(
    run_id: str,
    samples: list[dict],
    predictions: Mapping[str, list[float]],
    head_specs: list[BaselineHeadSpec] | tuple[BaselineHeadSpec, ...],
    model_family: str,
    aggregation_scope: str,
    loss_masks: Mapping[str, list[bool]] | None = None,
) -> list[dict]:
    """Convert per-target predictions into D1 long-format rows."""

    rows: list[dict] = []
    for spec in head_specs:
        if spec.name == "ops":
            raise ValueError("OPS must not be emitted as an event-level visual head")
        y_preds = predictions.get(spec.name)
        if y_preds is None:
            continue
        if len(y_preds) != len(samples):
            raise ValueError(f"prediction length mismatch for target {spec.name}")
        masks = loss_masks.get(spec.name) if loss_masks is not None else None
        for index, (sample, y_pred) in enumerate(zip(samples, y_preds)):
            available = bool(masks[index]) if masks is not None else sample.get(spec.column) is not None
            label_missing_reason = None if available else sample.get(
                f"{spec.name}_missing_reason",
                "label_missing",
            )
            rows.append(
                {
                    "run_id": run_id,
                    "sample_id": sample["sample_id"],
                    "event_id": sample["event_id"],
                    "batter_season_id": sample["batter_season_id"],
                    "prediction_level": "event",
                    "target_name": spec.name,
                    "y_true": sample.get(spec.column) if available else None,
                    "y_pred": float(y_pred) if available else None,
                    "target_available": available,
                    "target_source": spec.column,
                    "head_kind": spec.kind,
                    "loss_name": spec.loss,
                    "aggregation_scope": aggregation_scope,
                    "prior_mode": "none",
                    "label_missing_reason": label_missing_reason,
                    "requires_pa_manifest": spec.requires_pa_manifest,
                    "n_prior_clips": 0,
                    "aggregation_method": model_family,
                    "same_event_ensemble": False,
                    "prediction_std": None,
                    "split": sample.get("split", "unknown"),
                }
            )
    return rows
