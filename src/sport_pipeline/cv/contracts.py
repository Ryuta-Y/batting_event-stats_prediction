"""CV preprocessing artifact contracts.

Agent B owns the lifecycle from video source evidence to candidate segments,
contact-aligned clips, and debug overlays. These schemas are intentionally
lightweight so they can be validated locally without video dependencies.
"""

from __future__ import annotations

from sport_pipeline.schemas.data_manifest import ColumnSpec, ManifestSchema


CV_CONTRACT_VERSION = "cv_preprocess_contract_v1"


CANDIDATE_SEGMENTS_SCHEMA = ManifestSchema(
    name="candidate_segments_v1",
    version=CV_CONTRACT_VERSION,
    artifact_path="clips/{run_id}/candidate_segments_v1.parquet",
    primary_key=("candidate_segment_id",),
    columns=(
        ColumnSpec("schema_version", "string"),
        ColumnSpec("candidate_segment_id", "string"),
        ColumnSpec("video_source_id", "string"),
        ColumnSpec("source_video_id", "string", nullable=True),
        ColumnSpec("event_id", "string"),
        ColumnSpec("same_event_group_id", "string"),
        ColumnSpec("view_id", "string"),
        ColumnSpec("game_pk", "int"),
        ColumnSpec("play_id", "string", nullable=True),
        ColumnSpec("batter_id", "int"),
        ColumnSpec("season", "int"),
        ColumnSpec("batter_season_id", "string"),
        ColumnSpec("source_kind", "string"),
        ColumnSpec("source_topic", "string"),
        ColumnSpec("dataset_role", "string"),
        ColumnSpec("segment_kind", "string"),
        ColumnSpec("lifecycle_stage", "string"),
        ColumnSpec("review_status", "string"),
        ColumnSpec("status_reason", "string", nullable=True),
        ColumnSpec("start_frame", "int"),
        ColumnSpec("end_frame", "int"),
        ColumnSpec("start_time_sec", "float"),
        ColumnSpec("end_time_sec", "float"),
        ColumnSpec("fps", "float"),
        ColumnSpec("duration_sec", "float"),
        ColumnSpec("shot_change_score", "float"),
        ColumnSpec("shot_type", "string"),
        ColumnSpec("view_label", "string"),
        ColumnSpec("view_confidence", "float"),
        ColumnSpec("camera_motion_level", "string"),
        ColumnSpec("batter_visible", "bool"),
        ColumnSpec("batter_visibility_score", "float"),
        ColumnSpec("bat_visible", "bool"),
        ColumnSpec("bat_visibility_score", "float"),
        ColumnSpec("plate_visible", "bool"),
        ColumnSpec("plate_visibility_score", "float"),
        ColumnSpec("contact_visible", "bool"),
        ColumnSpec("contact_frame", "int", nullable=True),
        ColumnSpec("contact_time_sec", "float", nullable=True),
        ColumnSpec("contact_confidence", "float"),
        ColumnSpec("is_replay", "bool"),
        ColumnSpec("is_non_batting_segment", "bool"),
        ColumnSpec("is_dugout", "bool"),
        ColumnSpec("is_broadcast_cutaway", "bool"),
        ColumnSpec("wrong_event", "bool"),
        ColumnSpec("difficult_pitch", "bool"),
        ColumnSpec("extreme_form", "bool"),
        ColumnSpec("hard_negative", "bool"),
        ColumnSpec("trim_policy", "string"),
        ColumnSpec("trim_start_time_sec", "float", nullable=True),
        ColumnSpec("trim_end_time_sec", "float", nullable=True),
        ColumnSpec("quality_flags", "json"),
        ColumnSpec("outlier_flags", "json"),
    ),
)


CLIPS_SCHEMA = ManifestSchema(
    name="clips_v1",
    version=CV_CONTRACT_VERSION,
    artifact_path="clips/{run_id}/clips_v1.parquet",
    primary_key=("clip_id",),
    columns=(
        ColumnSpec("schema_version", "string"),
        ColumnSpec("clip_id", "string"),
        ColumnSpec("candidate_segment_id", "string"),
        ColumnSpec("video_source_id", "string"),
        ColumnSpec("event_id", "string"),
        ColumnSpec("same_event_group_id", "string"),
        ColumnSpec("view_id", "string"),
        ColumnSpec("crop_id", "string"),
        ColumnSpec("augmentation_id", "string"),
        ColumnSpec("game_pk", "int"),
        ColumnSpec("play_id", "string", nullable=True),
        ColumnSpec("batter_id", "int"),
        ColumnSpec("season", "int"),
        ColumnSpec("batter_season_id", "string"),
        ColumnSpec("clip_version", "string"),
        ColumnSpec("clip_status", "string"),
        ColumnSpec("quality_tier", "string"),
        ColumnSpec("dataset_role", "string"),
        ColumnSpec("cohort_role", "string"),
        ColumnSpec("clip_path", "string", nullable=True),
        ColumnSpec("overlay_path", "string", nullable=True),
        ColumnSpec("debug_frame_path", "string", nullable=True),
        ColumnSpec("start_frame", "int"),
        ColumnSpec("end_frame", "int"),
        ColumnSpec("start_time_sec", "float"),
        ColumnSpec("end_time_sec", "float"),
        ColumnSpec("fps", "float"),
        ColumnSpec("duration_sec", "float"),
        ColumnSpec("contact_frame", "int", nullable=True),
        ColumnSpec("contact_time_sec", "float", nullable=True),
        ColumnSpec("contact_confidence", "float"),
        ColumnSpec("contact_window_policy", "string"),
        ColumnSpec("pre_contact_sec", "float"),
        ColumnSpec("post_contact_sec", "float"),
        ColumnSpec("view_label", "string"),
        ColumnSpec("view_confidence", "float"),
        ColumnSpec("batter_visible", "bool"),
        ColumnSpec("batter_visibility_score", "float"),
        ColumnSpec("bat_visible", "bool"),
        ColumnSpec("bat_visibility_score", "float"),
        ColumnSpec("plate_visible", "bool"),
        ColumnSpec("plate_visibility_score", "float"),
        ColumnSpec("clean_cohort_eligible", "bool"),
        ColumnSpec("robustness_cohort_eligible", "bool"),
        ColumnSpec("hard_negative", "bool"),
        ColumnSpec("difficult_pitch", "bool"),
        ColumnSpec("extreme_form", "bool"),
        ColumnSpec("review_reason", "string", nullable=True),
        ColumnSpec("exclusion_reason", "string", nullable=True),
        ColumnSpec("target_alignment_status", "string"),
        ColumnSpec("join_key_fields", "json"),
        ColumnSpec("quality_flags", "json"),
        ColumnSpec("outlier_flags", "json"),
    ),
)


DEBUG_OVERLAYS_SCHEMA = ManifestSchema(
    name="debug_overlays_v1",
    version=CV_CONTRACT_VERSION,
    artifact_path="debug/{run_id}/debug_overlays_v1.parquet",
    primary_key=("debug_artifact_id",),
    columns=(
        ColumnSpec("schema_version", "string"),
        ColumnSpec("debug_artifact_id", "string"),
        ColumnSpec("clip_id", "string"),
        ColumnSpec("event_id", "string"),
        ColumnSpec("same_event_group_id", "string"),
        ColumnSpec("artifact_kind", "string"),
        ColumnSpec("artifact_path", "string"),
        ColumnSpec("frame_number", "int", nullable=True),
        ColumnSpec("time_sec", "float", nullable=True),
        ColumnSpec("view_label", "string"),
        ColumnSpec("quality_tier", "string"),
        ColumnSpec("created_by", "string"),
        ColumnSpec("includes_detection", "bool"),
        ColumnSpec("includes_tracking", "bool"),
        ColumnSpec("includes_pose", "bool"),
        ColumnSpec("includes_bat", "bool"),
        ColumnSpec("includes_plate", "bool"),
        ColumnSpec("includes_contact_window", "bool"),
        ColumnSpec("review_priority", "int"),
    ),
)


CV_SCHEMAS = {
    schema.name: schema
    for schema in (
        CANDIDATE_SEGMENTS_SCHEMA,
        CLIPS_SCHEMA,
        DEBUG_OVERLAYS_SCHEMA,
    )
}

