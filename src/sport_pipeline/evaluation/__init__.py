"""Evaluation contracts and helpers."""

from sport_pipeline.evaluation.evaluator import evaluate_predictions
from sport_pipeline.evaluation.predictions import PREDICTIONS_SCHEMA, validate_prediction_rows
from sport_pipeline.evaluation.target_registry import TargetSpec, load_target_registry

__all__ = [
    "PREDICTIONS_SCHEMA",
    "TargetSpec",
    "evaluate_predictions",
    "load_target_registry",
    "validate_prediction_rows",
]
