"""Minimal context-only baseline interface for D1.

This is intentionally tiny: it provides a deterministic local baseline and a
stable output shape while heavy CatBoost/XGBoost training remains Colab-side.
"""

from __future__ import annotations

from statistics import mean
from typing import Any

from sport_pipeline.evaluation.target_registry import TargetSpec


class ConstantContextBaseline:
    """Predict each supported target with its training-set mean label."""

    def __init__(self, targets: dict[str, TargetSpec], model_family: str = "context_constant_mean") -> None:
        self.targets = targets
        self.model_family = model_family
        self._means: dict[str, float] = {}

    def _records_for_target(
        self,
        records: list[dict[str, Any]],
        target: TargetSpec,
    ) -> list[dict[str, Any]]:
        if target.level != "player_season":
            return records

        by_batter_season: dict[str, dict[str, Any]] = {}
        for record in records:
            key = str(record.get("batter_season_id") or "")
            if not key:
                continue
            current = by_batter_season.get(key)
            if current is None or (
                current.get(target.column) is None and record.get(target.column) is not None
            ):
                by_batter_season[key] = record
        return list(by_batter_season.values())

    def fit(self, records: list[dict[str, Any]]) -> "ConstantContextBaseline":
        for name, target in self.targets.items():
            values = [
                float(record[target.column])
                for record in self._records_for_target(records, target)
                if target.column in record and record[target.column] is not None
            ]
            if values:
                self._means[name] = mean(values)
        return self

    def predict_rows(
        self,
        records: list[dict[str, Any]],
        run_id: str,
        split: str | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name, target in self.targets.items():
            for record in self._records_for_target(records, target):
                y_true = record.get(target.column)
                has_prediction = name in self._means
                target_available = y_true is not None and has_prediction
                missing_reason = None
                if y_true is None:
                    if target.requires_pa_manifest:
                        missing_reason = record.get("target_ops_missing_reason") or "pa_manifest_unavailable"
                    else:
                        missing_reason = record.get("label_missing_reason") or "label_missing"
                elif not has_prediction:
                    missing_reason = "baseline_not_fit_for_target"
                batter_season_id = str(record["batter_season_id"])
                event_id = None if target.level == "player_season" else record.get("event_id")
                sample_id = (
                    f"{batter_season_id}__{name}"
                    if target.level == "player_season"
                    else str(record.get("sample_id") or event_id or batter_season_id)
                )
                rows.append(
                    {
                        "run_id": run_id,
                        "sample_id": sample_id,
                        "event_id": event_id,
                        "batter_season_id": batter_season_id,
                        "prediction_level": target.level,
                        "target_name": name,
                        "y_true": y_true,
                        "y_pred": self._means.get(name),
                        "target_available": target_available,
                        "target_source": target.column,
                        "head_kind": target.kind,
                        "loss_name": target.loss,
                        "aggregation_scope": "context_only",
                        "prior_mode": "none",
                        "label_missing_reason": missing_reason,
                        "requires_pa_manifest": target.requires_pa_manifest,
                        "n_prior_clips": 0,
                        "aggregation_method": "none",
                        "same_event_ensemble": False,
                        "prediction_std": None,
                        "split": split,
                        "model_family": self.model_family,
                    }
                )
        return rows
