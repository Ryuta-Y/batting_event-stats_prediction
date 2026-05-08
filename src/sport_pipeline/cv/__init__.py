"""CV preprocessing contracts and metadata helpers."""

from sport_pipeline.cv.contracts import (
    CANDIDATE_SEGMENTS_SCHEMA,
    CLIPS_SCHEMA,
    CV_CONTRACT_VERSION,
    DEBUG_OVERLAYS_SCHEMA,
)
from sport_pipeline.cv.artifact_schemas import (
    BAT_LINES_SCHEMA,
    DETECTIONS_SCHEMA,
    HOMOGRAPHY_SCHEMA,
    OBJECTS_SCHEMA,
    POSE2D_SCHEMA,
    RAW_CV_SCHEMAS,
    TRACKS_SCHEMA,
)
from sport_pipeline.cv.quality import (
    CLEAN_CLIP_STATUS,
    EXCLUDED_STATUS,
    REVIEW_ONLY_STATUS,
    build_clip_id,
    classify_candidate_segment,
    derive_clip_metadata,
)
from sport_pipeline.cv.diagnostics import (
    clip_quality_diagnostics,
    format_clip_quality_diagnostics,
    is_clean_trainable_clip,
)
from sport_pipeline.cv.overlay_video import render_cv_overlay_videos

__all__ = [
    "CANDIDATE_SEGMENTS_SCHEMA",
    "CLIPS_SCHEMA",
    "CV_CONTRACT_VERSION",
    "DEBUG_OVERLAYS_SCHEMA",
    "DETECTIONS_SCHEMA",
    "TRACKS_SCHEMA",
    "POSE2D_SCHEMA",
    "OBJECTS_SCHEMA",
    "BAT_LINES_SCHEMA",
    "HOMOGRAPHY_SCHEMA",
    "RAW_CV_SCHEMAS",
    "CLEAN_CLIP_STATUS",
    "REVIEW_ONLY_STATUS",
    "EXCLUDED_STATUS",
    "build_clip_id",
    "classify_candidate_segment",
    "derive_clip_metadata",
    "clip_quality_diagnostics",
    "format_clip_quality_diagnostics",
    "is_clean_trainable_clip",
    "render_cv_overlay_videos",
]
