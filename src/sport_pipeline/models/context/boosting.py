"""Optional boosted context-only baseline for Colab runs.

The shipped smoke baseline stays dependency-free.  This module is imported only
when a user explicitly asks for the stronger Colab context baseline, and it
prefers CatBoost because the context features are a mix of numeric and
categorical baseball metadata.
"""

from __future__ import annotations

import math
from statistics import mean
from typing import Any

from sport_pipeline.evaluation.target_registry import TargetSpec


DEFAULT_CONTEXT_FEATURE_COLUMNS = (
    "batter_id",
    "pitcher_id",
    "season",
    "stand",
    "p_throws",
    "pitch_type",
    "release_speed",
    "plate_x",
    "plate_z",
    "zone",
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "inning_topbot",
    "home_team",
    "away_team",
)

EVENT_TARGETS = ("ev", "la", "hard_hit", "barrel", "xba", "xwoba")
DEFAULT_CATEGORICAL_FEATURES = (
    "batter_id",
    "pitcher_id",
    "season",
    "stand",
    "p_throws",
    "pitch_type",
    "zone",
    "inning_topbot",
    "home_team",
    "away_team",
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _target_missing_reason(record: dict[str, Any], target_name: str, target: TargetSpec) -> str:
    if target.requires_pa_manifest:
        return str(record.get("target_ops_missing_reason") or "pa_manifest_unavailable")
    return str(
        record.get(f"target_{target_name}_missing_reason")
        or record.get("label_missing_reason")
        or "label_missing"
    )


class CatBoostContextBaseline:
    """Per-target CatBoost models with predictions_v1 row output.

    This is a real supervised context baseline, but it is intentionally optional:
    CatBoost is imported lazily so local unit tests and smoke runs do not download
    or require heavyweight ML packages.  It trains only event-level targets from
    the target registry; player-season aggregate targets such as OPS are emitted
    once per batter-season as unavailable unless an upstream PA aggregate model
    is added later.
    """

    def __init__(
        self,
        targets: dict[str, TargetSpec],
        *,
        feature_columns: tuple[str, ...] = DEFAULT_CONTEXT_FEATURE_COLUMNS,
        categorical_features: tuple[str, ...] = DEFAULT_CATEGORICAL_FEATURES,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        depth: int = 6,
        random_seed: int = 2026,
        early_stopping_rounds: int = 50,
        validation_fraction: float = 0.2,
        task_type: str = "CPU",
        devices: str | None = None,
        model_family: str = "context_catboost",
        progress_callback: Any | None = None,
    ) -> None:
        self.targets = targets
        self.feature_columns = feature_columns
        self.categorical_features = categorical_features
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_seed = random_seed
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.task_type = task_type
        self.devices = devices
        self.model_family = model_family
        self.progress_callback = progress_callback
        self._models: dict[str, Any] = {}
        self._fallback_means: dict[str, float] = {}
        self._cat_feature_indices = [
            index for index, column in enumerate(self.feature_columns) if column in self.categorical_features
        ]

    def _feature_matrix(self, records: list[dict[str, Any]]) -> list[list[Any]]:
        matrix: list[list[Any]] = []
        for record in records:
            values: list[Any] = []
            for column in self.feature_columns:
                value = record.get(column)
                if column in self.categorical_features:
                    values.append("__MISSING__" if _is_missing(value) else str(value))
                else:
                    values.append(None if _is_missing(value) else float(value))
            matrix.append(values)
        return matrix

    def _fit_target(self, name: str, target: TargetSpec, records: list[dict[str, Any]]) -> None:
        usable = [record for record in records if not _is_missing(record.get(target.column))]
        if not usable:
            return
        values = [float(record[target.column]) for record in usable]
        self._fallback_means[name] = mean(values)
        if len(usable) < 4:
            return

        try:
            from catboost import CatBoostClassifier, CatBoostRegressor, Pool  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "CatBoost is required for --model-family catboost. In Colab run "
                "`pip install -q catboost` before this command, or use the default "
                "constant_mean baseline."
            ) from exc

        train_rows = usable
        eval_rows: list[dict[str, Any]] = []
        if 0.0 < self.validation_fraction < 0.5 and len(usable) >= 10:
            split_at = max(1, int(round(len(usable) * (1.0 - self.validation_fraction))))
            train_rows = usable[:split_at]
            eval_rows = usable[split_at:]

        train_pool = Pool(
            self._feature_matrix(train_rows),
            label=[float(record[target.column]) for record in train_rows],
            cat_features=self._cat_feature_indices,
        )
        eval_set = None
        if eval_rows:
            eval_set = Pool(
                self._feature_matrix(eval_rows),
                label=[float(record[target.column]) for record in eval_rows],
                cat_features=self._cat_feature_indices,
            )

        common_params = {
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "random_seed": self.random_seed,
            "allow_writing_files": False,
            "verbose": False,
            "task_type": self.task_type,
        }
        if self.devices:
            common_params["devices"] = self.devices
        if target.kind == "binary":
            model = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                **common_params,
            )
        else:
            model = CatBoostRegressor(
                loss_function="RMSE",
                eval_metric="RMSE",
                **common_params,
            )
        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["early_stopping_rounds"] = self.early_stopping_rounds
        model.fit(train_pool, **fit_kwargs)
        self._models[name] = model

    def fit(self, records: list[dict[str, Any]]) -> "CatBoostContextBaseline":
        event_targets = {
            name: target
            for name, target in self.targets.items()
            if target.level == "event" and name in EVENT_TARGETS
        }
        for name, target in event_targets.items():
            if self.progress_callback is not None:
                self.progress_callback("training_target", target_name=name)
            self._fit_target(name, target, records)
            if self.progress_callback is not None:
                self.progress_callback(
                    "target_complete",
                    target_name=name,
                    fitted=name in self._models,
                    fallback_available=name in self._fallback_means,
                )
        return self

    def _predict_value(self, name: str, target: TargetSpec, record: dict[str, Any]) -> float | None:
        model = self._models.get(name)
        if model is None:
            return self._fallback_means.get(name)
        features = self._feature_matrix([record])
        if target.kind == "binary":
            return float(model.predict_proba(features)[0][1])
        return float(model.predict(features)[0])

    def _player_season_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_batter_season: dict[str, dict[str, Any]] = {}
        for record in records:
            key = str(record.get("batter_season_id") or "")
            if key and key not in by_batter_season:
                by_batter_season[key] = record
        return list(by_batter_season.values())

    def predict_rows(
        self,
        records: list[dict[str, Any]],
        run_id: str,
        split: str | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name, target in self.targets.items():
            target_records = records if target.level == "event" else self._player_season_records(records)
            for record in target_records:
                y_true = record.get(target.column)
                y_pred = None
                target_available = False
                missing_reason = None
                if target.level == "event" and name in EVENT_TARGETS and not _is_missing(y_true):
                    y_pred = self._predict_value(name, target, record)
                    target_available = y_pred is not None
                    if not target_available:
                        missing_reason = "context_catboost_not_fit_for_target"
                elif _is_missing(y_true):
                    missing_reason = _target_missing_reason(record, name, target)
                else:
                    missing_reason = "player_season_context_model_not_implemented"

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
                        "y_true": None if _is_missing(y_true) else y_true,
                        "y_pred": y_pred,
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
