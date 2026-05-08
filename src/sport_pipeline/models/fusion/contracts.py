"""D3 fusion-owned audit contracts."""

from __future__ import annotations

from sport_pipeline.schemas.data_manifest import ColumnSpec, ManifestSchema


FUSION_CONTRACT_VERSION = "fusion_contract_v1"


FUSION_INPUT_AUDIT_SCHEMA = ManifestSchema(
    name="fusion_input_audit_v1",
    version=FUSION_CONTRACT_VERSION,
    artifact_path="predictions/{run_id}/fusion_input_audit_v1.parquet",
    primary_key=("fusion_sample_id", "source_sample_id", "source_target_name"),
    columns=(
        ColumnSpec("schema_version", "string"),
        ColumnSpec("fusion_run_id", "string"),
        ColumnSpec("fusion_sample_id", "string"),
        ColumnSpec("source_run_id", "string"),
        ColumnSpec("source_sample_id", "string"),
        ColumnSpec("source_event_id", "string", nullable=True),
        ColumnSpec("source_batter_season_id", "string"),
        ColumnSpec("source_prediction_level", "string"),
        ColumnSpec("source_target_name", "string"),
        ColumnSpec("source_aggregation_scope", "string"),
        ColumnSpec("source_prior_mode", "string"),
        ColumnSpec("source_same_event_ensemble", "bool"),
        ColumnSpec("source_n_prior_clips", "int"),
        ColumnSpec("source_prediction_std", "float", nullable=True),
        ColumnSpec("source_target_available", "bool"),
        ColumnSpec("source_label_missing_reason", "string", nullable=True),
        ColumnSpec("fusion_weight", "float"),
    ),
)

