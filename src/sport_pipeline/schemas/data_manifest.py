"""Data manifest schema contracts owned by Data Agent A.

These definitions intentionally stay lightweight: they are enough for local
contract tests and Colab preflight checks, while full parquet IO remains a
later implementation detail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str
    required: bool = True
    nullable: bool = False
    description: str = ""


@dataclass(frozen=True)
class ManifestSchema:
    name: str
    version: str
    artifact_path: str
    columns: tuple[ColumnSpec, ...]
    primary_key: tuple[str, ...]

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns if column.required)

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    def column(self, name: str) -> ColumnSpec:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(name)


COMMON_METADATA = (
    ColumnSpec("schema_version", "string", description="Contract version."),
)


BBE_EVENTS_SCHEMA = ManifestSchema(
    name="bbe_events_v1",
    version="data_manifest_contract_v1",
    artifact_path="manifests/bbe_events_v1.parquet",
    primary_key=("event_id",),
    columns=COMMON_METADATA
    + (
        ColumnSpec("event_id", "string", description="Stable BBE event id."),
        ColumnSpec("game_pk", "int", description="MLB game key."),
        ColumnSpec("game_date", "date", description="YYYY-MM-DD event date."),
        ColumnSpec("season", "int", description="Season year."),
        ColumnSpec("batter_id", "int", description="Normalized batter id."),
        ColumnSpec("pitcher_id", "int", description="Normalized pitcher id."),
        ColumnSpec("batter_season_id", "string", description="{batter_id}_{season}."),
        ColumnSpec("at_bat_number", "int", nullable=True),
        ColumnSpec("pitch_number", "int", nullable=True),
        ColumnSpec("play_id", "string", nullable=True),
        ColumnSpec("same_event_group_id", "string", description="Same-event view/crop/augmentation group."),
        ColumnSpec("player_name", "string"),
        ColumnSpec("events", "string"),
        ColumnSpec("description", "string"),
        ColumnSpec("bb_type", "string", nullable=True),
        ColumnSpec("launch_speed", "float", nullable=True),
        ColumnSpec("launch_angle", "float", nullable=True),
        ColumnSpec("launch_speed_angle", "int", nullable=True),
        ColumnSpec("estimated_ba_using_speedangle", "float", nullable=True),
        ColumnSpec("estimated_woba_using_speedangle", "float", nullable=True),
        ColumnSpec("stand", "string", nullable=True),
        ColumnSpec("p_throws", "string", nullable=True),
        ColumnSpec("pitch_type", "string", nullable=True),
        ColumnSpec("release_speed", "float", nullable=True),
        ColumnSpec("plate_x", "float", nullable=True),
        ColumnSpec("plate_z", "float", nullable=True),
        ColumnSpec("zone", "int", nullable=True),
        ColumnSpec("balls", "int", nullable=True),
        ColumnSpec("strikes", "int", nullable=True),
        ColumnSpec("outs_when_up", "int", nullable=True),
        ColumnSpec("inning", "int", nullable=True),
        ColumnSpec("inning_topbot", "string", nullable=True),
        ColumnSpec("home_team", "string"),
        ColumnSpec("away_team", "string"),
        ColumnSpec("sv_id", "string", nullable=True),
        ColumnSpec("is_bbe", "bool"),
        ColumnSpec("is_home_run", "bool"),
        ColumnSpec("dataset_role", "string"),
        ColumnSpec("outcome_bin", "string"),
        ColumnSpec("ev_bin", "string"),
        ColumnSpec("la_bin", "string"),
        ColumnSpec("bb_type_bin", "string", nullable=True),
        ColumnSpec("has_video_candidate", "bool"),
        ColumnSpec("n_video_candidates", "int"),
        ColumnSpec("video_availability_score", "float"),
        ColumnSpec("target_ev_available", "bool"),
        ColumnSpec("target_la_available", "bool"),
        ColumnSpec("target_hard_hit_available", "bool"),
        ColumnSpec("target_barrel_available", "bool"),
        ColumnSpec("target_xba_available", "bool"),
        ColumnSpec("target_xwoba_available", "bool"),
        ColumnSpec("target_ops_available", "bool"),
        ColumnSpec("target_ops_missing_reason", "string", nullable=True),
        ColumnSpec("label_missing_reason", "string", nullable=True),
        ColumnSpec("clean_location_cohort_v1", "bool"),
        ColumnSpec("clean_count_cohort_v1", "bool"),
        ColumnSpec("usable_for_event_model", "bool"),
        ColumnSpec("quality_flags", "json"),
        ColumnSpec("outlier_flags", "json"),
        ColumnSpec("review_status", "string"),
        ColumnSpec("reject_reason", "string", nullable=True),
    ),
)


VIDEO_SOURCES_SCHEMA = ManifestSchema(
    name="video_sources_v1",
    version="data_manifest_contract_v1",
    artifact_path="manifests/video_sources_v1.parquet",
    primary_key=("video_source_id",),
    columns=COMMON_METADATA
    + (
        ColumnSpec("video_source_id", "string"),
        ColumnSpec("event_id", "string"),
        ColumnSpec("same_event_group_id", "string"),
        ColumnSpec("source_video_id", "string", nullable=True),
        ColumnSpec("view_id", "string"),
        ColumnSpec("source_kind", "string"),
        ColumnSpec("source_url", "string", nullable=True),
        ColumnSpec("media_url", "string", nullable=True),
        ColumnSpec("source_topic", "string"),
        ColumnSpec("dataset_role", "string"),
        ColumnSpec("rights_status", "string"),
        ColumnSpec("match_confidence", "float"),
        ColumnSpec("match_reason", "string"),
        ColumnSpec("join_key_fields", "json"),
        ColumnSpec("candidate_rank", "int"),
        ColumnSpec("video_available", "bool"),
        ColumnSpec("download_status", "string"),
        ColumnSpec("local_video_path", "string", nullable=True),
        ColumnSpec("probe_status", "string"),
        ColumnSpec("review_status", "string"),
        ColumnSpec("reject_reason", "string", nullable=True),
        ColumnSpec("view_label", "string"),
        ColumnSpec("view_confidence", "float"),
        ColumnSpec("batting_visibility", "string"),
        ColumnSpec("is_replay", "bool"),
        ColumnSpec("is_non_batting_segment", "bool"),
    ),
)


PLAYER_GROUP_SPLIT_SCHEMA = ManifestSchema(
    name="player_group_split_v1",
    version="data_manifest_contract_v1",
    artifact_path="manifests/splits/player_group_split_v1.parquet",
    primary_key=("event_id",),
    columns=COMMON_METADATA
    + (
        ColumnSpec("event_id", "string"),
        ColumnSpec("batter_id", "int"),
        ColumnSpec("season", "int"),
        ColumnSpec("batter_season_id", "string"),
        ColumnSpec("split", "string"),
        ColumnSpec("split_strategy", "string"),
        ColumnSpec("group_key", "string"),
        ColumnSpec("created_at", "string"),
    ),
)


TEMPORAL_SPLIT_SCHEMA = ManifestSchema(
    name="temporal_split_v1",
    version="data_manifest_contract_v1",
    artifact_path="manifests/splits/temporal_split_v1.parquet",
    primary_key=("event_id",),
    columns=COMMON_METADATA
    + (
        ColumnSpec("event_id", "string"),
        ColumnSpec("batter_id", "int"),
        ColumnSpec("season", "int"),
        ColumnSpec("batter_season_id", "string"),
        ColumnSpec("game_date", "date"),
        ColumnSpec("split", "string"),
        ColumnSpec("split_strategy", "string"),
        ColumnSpec("cutoff_date", "date"),
        ColumnSpec("created_at", "string"),
    ),
)


SCHEMAS = {
    schema.name: schema
    for schema in (
        BBE_EVENTS_SCHEMA,
        VIDEO_SOURCES_SCHEMA,
        PLAYER_GROUP_SPLIT_SCHEMA,
        TEMPORAL_SPLIT_SCHEMA,
    )
}


PYTHON_TYPES = {
    "string": str,
    "int": int,
    "float": (int, float),
    "bool": bool,
    "date": str,
    "json": (list, dict, str),
}


class SchemaValidationError(ValueError):
    """Raised when rows violate a manifest schema contract."""


def _is_integer_valued_float(value: Any) -> bool:
    """Return true for parquet/pandas nullable integers materialized as floats."""

    return isinstance(value, float) and value.is_integer()


def validate_rows(schema: ManifestSchema, rows: Iterable[dict[str, Any]]) -> None:
    """Validate required columns and simple scalar types for manifest rows."""

    for index, row in enumerate(rows):
        missing = [name for name in schema.required_columns if name not in row]
        if missing:
            raise SchemaValidationError(f"{schema.name} row {index} missing columns: {missing}")
        for column in schema.columns:
            if column.name not in row:
                continue
            value = row[column.name]
            if value is None:
                if column.nullable:
                    continue
                raise SchemaValidationError(f"{schema.name} row {index} has null {column.name}")
            expected = PYTHON_TYPES[column.dtype]
            if column.dtype == "float" and isinstance(value, bool):
                raise SchemaValidationError(f"{schema.name} row {index} has bool for float {column.name}")
            if column.dtype == "int" and isinstance(value, bool):
                raise SchemaValidationError(f"{schema.name} row {index} has bool for int {column.name}")
            if column.dtype == "int" and _is_integer_valued_float(value):
                continue
            if not isinstance(value, expected):
                raise SchemaValidationError(
                    f"{schema.name} row {index} column {column.name} expected {column.dtype}, "
                    f"got {type(value).__name__}"
                )
