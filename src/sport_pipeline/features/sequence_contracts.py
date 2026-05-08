"""Structured sequence and embedding artifact contracts."""

from __future__ import annotations

from sport_pipeline.schemas.data_manifest import ColumnSpec, ManifestSchema


SEQUENCE_CONTRACT_VERSION = "sequence_contract_v1"


STRUCTURED_FRAME_FEATURES_SCHEMA = ManifestSchema(
    name="structured_frame_features_v1",
    version=SEQUENCE_CONTRACT_VERSION,
    artifact_path="features/structured_sequence_v1/frames.parquet",
    primary_key=("sequence_id", "frame_index"),
    columns=(
        ColumnSpec("schema_version", "string"),
        ColumnSpec("sequence_id", "string"),
        ColumnSpec("clip_id", "string"),
        ColumnSpec("event_id", "string"),
        ColumnSpec("same_event_group_id", "string"),
        ColumnSpec("view_id", "string"),
        ColumnSpec("batter_id", "int"),
        ColumnSpec("season", "int"),
        ColumnSpec("batter_season_id", "string"),
        ColumnSpec("frame_index", "int"),
        ColumnSpec("time_sec", "float"),
        ColumnSpec("relative_time_to_contact_sec", "float", nullable=True),
        ColumnSpec("phase_label", "string"),
        ColumnSpec("phase_confidence", "float"),
        ColumnSpec("feature_namespace", "string"),
        ColumnSpec("feature_names", "json"),
        ColumnSpec("feature_values", "json"),
        ColumnSpec("feature_mask", "json"),
        ColumnSpec("view_label", "string"),
        ColumnSpec("quality_tier", "string"),
        ColumnSpec("clip_status", "string"),
    ),
)


STRUCTURED_SEQUENCE_MANIFEST_SCHEMA = ManifestSchema(
    name="structured_sequence_manifest_v1",
    version=SEQUENCE_CONTRACT_VERSION,
    artifact_path="features/structured_sequence_v1/manifest.parquet",
    primary_key=("sequence_id",),
    columns=(
        ColumnSpec("schema_version", "string"),
        ColumnSpec("sequence_id", "string"),
        ColumnSpec("sample_id", "string"),
        ColumnSpec("clip_id", "string"),
        ColumnSpec("event_id", "string"),
        ColumnSpec("same_event_group_id", "string"),
        ColumnSpec("view_id", "string"),
        ColumnSpec("batter_id", "int"),
        ColumnSpec("season", "int"),
        ColumnSpec("batter_season_id", "string"),
        ColumnSpec("game_date", "date"),
        ColumnSpec("split", "string"),
        ColumnSpec("sequence_path", "string", nullable=True),
        ColumnSpec("n_frames", "int"),
        ColumnSpec("feature_dim", "int"),
        ColumnSpec("feature_names", "json"),
        ColumnSpec("phase_labels", "json"),
        ColumnSpec("frame_rate", "float"),
        ColumnSpec("clip_version", "string"),
        ColumnSpec("clip_status", "string"),
        ColumnSpec("quality_tier", "string"),
        ColumnSpec("view_label", "string"),
        ColumnSpec("target_available", "bool"),
        ColumnSpec("target_missing_reason", "string", nullable=True),
    ),
)


CLIP_EMBEDDING_SCHEMA = ManifestSchema(
    name="clip_embedding_v1",
    version=SEQUENCE_CONTRACT_VERSION,
    artifact_path="features/clip_embedding_v1/manifest.parquet",
    primary_key=("clip_id", "encoder_name", "encoder_version"),
    columns=(
        ColumnSpec("schema_version", "string"),
        ColumnSpec("clip_id", "string"),
        ColumnSpec("sequence_id", "string"),
        ColumnSpec("event_id", "string"),
        ColumnSpec("same_event_group_id", "string"),
        ColumnSpec("view_id", "string"),
        ColumnSpec("batter_id", "int"),
        ColumnSpec("season", "int"),
        ColumnSpec("batter_season_id", "string"),
        ColumnSpec("game_date", "date"),
        ColumnSpec("split", "string"),
        ColumnSpec("encoder_name", "string"),
        ColumnSpec("encoder_version", "string"),
        ColumnSpec("embedding_path", "string", nullable=True),
        ColumnSpec("embedding_values", "json"),
        ColumnSpec("embedding_dim", "int"),
        ColumnSpec("quality_tier", "string"),
        ColumnSpec("clip_status", "string"),
        ColumnSpec("view_label", "string"),
        ColumnSpec("view_confidence", "float"),
        ColumnSpec("contact_confidence", "float"),
        ColumnSpec("pose_coverage", "float"),
        ColumnSpec("clean_cohort_eligible", "bool"),
        ColumnSpec("target_alignment_status", "string"),
    ),
)


PLAYER_SEASON_EMBEDDING_SCHEMA = ManifestSchema(
    name="player_season_embedding_v1",
    version=SEQUENCE_CONTRACT_VERSION,
    artifact_path="features/player_season_embedding_v1/manifest.parquet",
    primary_key=("batter_season_id", "prior_mode", "aggregation_method", "cutoff_date"),
    columns=(
        ColumnSpec("schema_version", "string"),
        ColumnSpec("batter_season_id", "string"),
        ColumnSpec("batter_id", "int"),
        ColumnSpec("season", "int"),
        ColumnSpec("aggregator_name", "string"),
        ColumnSpec("aggregator_version", "string"),
        ColumnSpec("aggregation_method", "string"),
        ColumnSpec("aggregation_scope", "string"),
        ColumnSpec("prior_mode", "string"),
        ColumnSpec("cutoff_event_id", "string", nullable=True),
        ColumnSpec("cutoff_date", "date", nullable=True),
        ColumnSpec("n_clips_total", "int"),
        ColumnSpec("n_clips_used", "int"),
        ColumnSpec("clip_ids_used", "json"),
        ColumnSpec("source_event_ids_used", "json"),
        ColumnSpec("uses_future_clips", "bool"),
        ColumnSpec("embedding_path", "string", nullable=True),
        ColumnSpec("embedding_values", "json"),
        ColumnSpec("embedding_dim", "int"),
        ColumnSpec("quality_policy", "string"),
        ColumnSpec("split_policy", "string"),
    ),
)


SEQUENCE_SCHEMAS = {
    schema.name: schema
    for schema in (
        STRUCTURED_FRAME_FEATURES_SCHEMA,
        STRUCTURED_SEQUENCE_MANIFEST_SCHEMA,
        CLIP_EMBEDDING_SCHEMA,
        PLAYER_SEASON_EMBEDDING_SCHEMA,
    )
}

