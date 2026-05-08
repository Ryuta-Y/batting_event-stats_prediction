"""Sequence model interfaces and aggregation helpers."""

from sport_pipeline.models.sequence.aggregation import (
    attention_pool,
    mean_pool,
    quality_weighted_pool,
    same_event_ensemble_predictions,
)
from sport_pipeline.models.sequence.interface import SequenceModelInterface, TCNBaselineConfig

__all__ = [
    "SequenceModelInterface",
    "TCNBaselineConfig",
    "attention_pool",
    "mean_pool",
    "quality_weighted_pool",
    "same_event_ensemble_predictions",
]

