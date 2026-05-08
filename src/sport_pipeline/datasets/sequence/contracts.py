"""Sequence dataset contracts, including event + player-season prior rows."""

from __future__ import annotations

from sport_pipeline.schemas.data_manifest import ColumnSpec, ManifestSchema


SEQUENCE_DATASET_CONTRACT_VERSION = "sequence_dataset_contract_v1"


SEQUENCE_DATASET_SCHEMA = ManifestSchema(
    name="sequence_dataset_v1",
    version=SEQUENCE_DATASET_CONTRACT_VERSION,
    artifact_path="datasets/sequence_dataset_v1/manifest.parquet",
    primary_key=("sample_id",),
    columns=(
        ColumnSpec("schema_version", "string"),
        ColumnSpec("sample_id", "string"),
        ColumnSpec("sequence_id", "string"),
        ColumnSpec("clip_id", "string"),
        ColumnSpec("event_id", "string"),
        ColumnSpec("same_event_group_id", "string"),
        ColumnSpec("batter_id", "int"),
        ColumnSpec("season", "int"),
        ColumnSpec("batter_season_id", "string"),
        ColumnSpec("game_date", "date"),
        ColumnSpec("split", "string"),
        ColumnSpec("sequence_path", "string", nullable=True),
        ColumnSpec("context_feature_path", "string", nullable=True),
        ColumnSpec("target_registry_version", "string"),
        ColumnSpec("prediction_level", "string"),
        ColumnSpec("aggregation_scope", "string"),
        ColumnSpec("prior_mode", "string"),
        ColumnSpec("n_prior_clips", "int"),
        ColumnSpec("quality_tier", "string"),
        ColumnSpec("view_label", "string"),
        ColumnSpec("target_available", "bool"),
        ColumnSpec("label_missing_reason", "string", nullable=True),
    ),
)


EVENT_WITH_PLAYER_PRIOR_SCHEMA = ManifestSchema(
    name="event_with_player_prior_v1",
    version=SEQUENCE_DATASET_CONTRACT_VERSION,
    artifact_path="datasets/event_with_player_prior_v1/manifest.parquet",
    primary_key=("sample_id", "prior_mode", "aggregation_method"),
    columns=(
        ColumnSpec("schema_version", "string"),
        ColumnSpec("sample_id", "string"),
        ColumnSpec("event_id", "string"),
        ColumnSpec("clip_id", "string"),
        ColumnSpec("sequence_id", "string"),
        ColumnSpec("same_event_group_id", "string"),
        ColumnSpec("batter_id", "int"),
        ColumnSpec("season", "int"),
        ColumnSpec("batter_season_id", "string"),
        ColumnSpec("game_date", "date"),
        ColumnSpec("split", "string"),
        ColumnSpec("context_feature_path", "string", nullable=True),
        ColumnSpec("current_clip_embedding_path", "string", nullable=True),
        ColumnSpec("current_clip_embedding_values", "json"),
        ColumnSpec("player_season_embedding_path", "string", nullable=True),
        ColumnSpec("player_season_embedding_values", "json"),
        ColumnSpec("prior_mode", "string"),
        ColumnSpec("aggregation_method", "string"),
        ColumnSpec("aggregation_scope", "string"),
        ColumnSpec("n_prior_clips", "int"),
        ColumnSpec("prior_clip_ids", "json"),
        ColumnSpec("uses_future_clips", "bool"),
        ColumnSpec("prediction_level", "string"),
        ColumnSpec("same_event_ensemble", "bool"),
        ColumnSpec("target_registry_version", "string"),
        ColumnSpec("target_ev_available", "bool"),
        ColumnSpec("target_la_available", "bool"),
        ColumnSpec("target_hard_hit_available", "bool"),
        ColumnSpec("target_barrel_available", "bool"),
        ColumnSpec("target_xba_available", "bool"),
        ColumnSpec("target_xwoba_available", "bool"),
        ColumnSpec("target_ops_available", "bool"),
        ColumnSpec("label_missing_reason", "string", nullable=True),
    ),
)


SEQUENCE_DATASET_SCHEMAS = {
    schema.name: schema
    for schema in (
        SEQUENCE_DATASET_SCHEMA,
        EVENT_WITH_PLAYER_PRIOR_SCHEMA,
    )
}

