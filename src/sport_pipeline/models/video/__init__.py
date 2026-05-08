"""Frozen video baseline interfaces."""

from sport_pipeline.models.video.heads import (
    BaselineHeadSpec,
    MultiTargetLightweightHead,
    build_event_head_specs,
    build_loss_masks,
)
from sport_pipeline.models.video.interface import FrozenVideoBaseline, FrozenVideoBaselineConfig
from sport_pipeline.models.video.predictions import build_visual_prediction_rows

__all__ = [
    "BaselineHeadSpec",
    "FrozenVideoBaseline",
    "FrozenVideoBaselineConfig",
    "MultiTargetLightweightHead",
    "build_event_head_specs",
    "build_loss_masks",
    "build_visual_prediction_rows",
]

