"""Context-only baseline interfaces."""

from sport_pipeline.models.context.baseline import ConstantContextBaseline
from sport_pipeline.models.context.boosting import (
    CatBoostContextBaseline,
    DEFAULT_CONTEXT_FEATURE_COLUMNS,
)

__all__ = ["CatBoostContextBaseline", "ConstantContextBaseline", "DEFAULT_CONTEXT_FEATURE_COLUMNS"]
