"""Prediction row schema validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class PredictionColumn:
    name: str
    dtype: str
    nullable: bool = False


PREDICTIONS_SCHEMA = (
    PredictionColumn("run_id", "string"),
    PredictionColumn("sample_id", "string"),
    PredictionColumn("event_id", "string", nullable=True),
    PredictionColumn("batter_season_id", "string"),
    PredictionColumn("prediction_level", "string"),
    PredictionColumn("target_name", "string"),
    PredictionColumn("y_true", "number", nullable=True),
    PredictionColumn("y_pred", "number", nullable=True),
    PredictionColumn("target_available", "bool"),
    PredictionColumn("target_source", "string"),
    PredictionColumn("head_kind", "string"),
    PredictionColumn("loss_name", "string"),
    PredictionColumn("aggregation_scope", "string"),
    PredictionColumn("prior_mode", "string"),
    PredictionColumn("label_missing_reason", "string", nullable=True),
    PredictionColumn("requires_pa_manifest", "bool", nullable=True),
    PredictionColumn("n_prior_clips", "int", nullable=True),
    PredictionColumn("aggregation_method", "string", nullable=True),
    PredictionColumn("same_event_ensemble", "bool", nullable=True),
    PredictionColumn("prediction_std", "number", nullable=True),
)


TYPE_MAP = {
    "string": str,
    "number": (int, float),
    "bool": bool,
    "int": int,
}


class PredictionValidationError(ValueError):
    """Raised when a prediction row violates the D1 contract."""


def _is_integer_valued_float(value: Any) -> bool:
    return isinstance(value, float) and value.is_integer()


def validate_prediction_rows(rows: Iterable[dict[str, Any]]) -> None:
    """Validate required predictions_v1 columns and simple scalar types."""

    for index, row in enumerate(rows):
        for column in PREDICTIONS_SCHEMA:
            if column.name not in row:
                if column.nullable:
                    continue
                raise PredictionValidationError(f"row {index} missing {column.name}")
            value = row[column.name]
            if value is None:
                if column.nullable:
                    continue
                raise PredictionValidationError(f"row {index} has null {column.name}")
            expected = TYPE_MAP[column.dtype]
            if column.dtype in {"number", "int"} and isinstance(value, bool):
                raise PredictionValidationError(f"row {index} has bool for {column.name}")
            if column.dtype == "int" and _is_integer_valued_float(value):
                continue
            if not isinstance(value, expected):
                raise PredictionValidationError(
                    f"row {index} column {column.name} expected {column.dtype}, "
                    f"got {type(value).__name__}"
                )
        target_available = row.get("target_available")
        if target_available is False and not row.get("label_missing_reason"):
            raise PredictionValidationError(
                f"row {index} unavailable target must include label_missing_reason"
            )
        if row.get("target_name") == "ops" and row.get("prediction_level") == "event":
            raise PredictionValidationError("OPS must not be emitted as event-level prediction")
