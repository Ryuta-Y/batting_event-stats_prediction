"""Raw CV artifact schemas for detector/pose/geometry outputs.

These contracts are intentionally dependency-free. The heavy YOLO/RTMPose
inference entrypoints write rows that follow these schemas, while local tests
can validate small synthetic rows without downloading model weights.
"""

from __future__ import annotations

from sport_pipeline.schemas.data_manifest import ColumnSpec, ManifestSchema


CV_ARTIFACT_CONTRACT_VERSION = "cv_artifact_contract_v1"


DETECTIONS_SCHEMA = ManifestSchema(
    name="detections_v1",
    version=CV_ARTIFACT_CONTRACT_VERSION,
    artifact_path="detections/{run_id}/detections_v1.parquet",
    primary_key=("detection_id",),
    columns=(
        ColumnSpec("schema_version", "string"),
        ColumnSpec("detection_id", "string"),
        ColumnSpec("clip_id", "string"),
        ColumnSpec("event_id", "string"),
        ColumnSpec("same_event_group_id", "string"),
        ColumnSpec("view_id", "string"),
        ColumnSpec("frame_index", "int"),
        ColumnSpec("time_sec", "float"),
        ColumnSpec("class_name", "string"),
        ColumnSpec("class_id", "int", nullable=True),
        ColumnSpec("confidence", "float"),
        ColumnSpec("x1", "float"),
        ColumnSpec("y1", "float"),
        ColumnSpec("x2", "float"),
        ColumnSpec("y2", "float"),
        ColumnSpec("track_id", "int", nullable=True),
        ColumnSpec("model_name", "string"),
        ColumnSpec("model_version", "string"),
        ColumnSpec("runtime", "string"),
    ),
)


TRACKS_SCHEMA = ManifestSchema(
    name="tracks_v1",
    version=CV_ARTIFACT_CONTRACT_VERSION,
    artifact_path="tracks/{run_id}/tracks_v1.parquet",
    primary_key=("track_row_id",),
    columns=(
        ColumnSpec("schema_version", "string"),
        ColumnSpec("track_row_id", "string"),
        ColumnSpec("clip_id", "string"),
        ColumnSpec("event_id", "string"),
        ColumnSpec("same_event_group_id", "string"),
        ColumnSpec("view_id", "string"),
        ColumnSpec("track_id", "int"),
        ColumnSpec("class_name", "string"),
        ColumnSpec("start_frame", "int"),
        ColumnSpec("end_frame", "int"),
        ColumnSpec("n_frames", "int"),
        ColumnSpec("mean_confidence", "float"),
        ColumnSpec("fragmentation", "float"),
        ColumnSpec("tracker_name", "string"),
        ColumnSpec("model_name", "string"),
    ),
)


POSE2D_SCHEMA = ManifestSchema(
    name="pose2d_v1",
    version=CV_ARTIFACT_CONTRACT_VERSION,
    artifact_path="pose2d/{run_id}/pose2d_v1.parquet",
    primary_key=("pose_id",),
    columns=(
        ColumnSpec("schema_version", "string"),
        ColumnSpec("pose_id", "string"),
        ColumnSpec("clip_id", "string"),
        ColumnSpec("event_id", "string"),
        ColumnSpec("same_event_group_id", "string"),
        ColumnSpec("view_id", "string"),
        ColumnSpec("frame_index", "int"),
        ColumnSpec("time_sec", "float"),
        ColumnSpec("track_id", "int", nullable=True),
        ColumnSpec("keypoint_schema", "string"),
        ColumnSpec("keypoints_xy", "json"),
        ColumnSpec("keypoint_scores", "json"),
        ColumnSpec("pose_score", "float"),
        ColumnSpec("model_name", "string"),
        ColumnSpec("model_version", "string"),
        ColumnSpec("runtime", "string"),
    ),
)


OBJECTS_SCHEMA = ManifestSchema(
    name="objects_v1",
    version=CV_ARTIFACT_CONTRACT_VERSION,
    artifact_path="objects/{run_id}/bat_detection_v1.parquet",
    primary_key=("object_id",),
    columns=(
        ColumnSpec("schema_version", "string"),
        ColumnSpec("object_id", "string"),
        ColumnSpec("clip_id", "string"),
        ColumnSpec("event_id", "string"),
        ColumnSpec("same_event_group_id", "string"),
        ColumnSpec("view_id", "string"),
        ColumnSpec("frame_index", "int"),
        ColumnSpec("time_sec", "float"),
        ColumnSpec("object_type", "string"),
        ColumnSpec("confidence", "float"),
        ColumnSpec("bbox_xyxy", "json"),
        ColumnSpec("mask_path", "string", nullable=True),
        ColumnSpec("source_detection_id", "string", nullable=True),
        ColumnSpec("model_name", "string"),
        ColumnSpec("model_version", "string"),
    ),
)


BAT_LINES_SCHEMA = ManifestSchema(
    name="bat_line_v1",
    version=CV_ARTIFACT_CONTRACT_VERSION,
    artifact_path="objects/{run_id}/bat_line_v1.parquet",
    primary_key=("bat_line_id",),
    columns=(
        ColumnSpec("schema_version", "string"),
        ColumnSpec("bat_line_id", "string"),
        ColumnSpec("clip_id", "string"),
        ColumnSpec("event_id", "string"),
        ColumnSpec("same_event_group_id", "string"),
        ColumnSpec("view_id", "string"),
        ColumnSpec("frame_index", "int"),
        ColumnSpec("time_sec", "float"),
        ColumnSpec("knob_x", "float"),
        ColumnSpec("knob_y", "float"),
        ColumnSpec("tip_x", "float"),
        ColumnSpec("tip_y", "float"),
        ColumnSpec("bat_angle_deg", "float"),
        ColumnSpec("bat_visible_length_px", "float"),
        ColumnSpec("confidence", "float"),
        ColumnSpec("source_object_id", "string", nullable=True),
        ColumnSpec("method", "string"),
    ),
)


HOMOGRAPHY_SCHEMA = ManifestSchema(
    name="homography_v1",
    version=CV_ARTIFACT_CONTRACT_VERSION,
    artifact_path="homography/{run_id}/homography_v1.parquet",
    primary_key=("homography_id",),
    columns=(
        ColumnSpec("schema_version", "string"),
        ColumnSpec("homography_id", "string"),
        ColumnSpec("clip_id", "string"),
        ColumnSpec("event_id", "string"),
        ColumnSpec("same_event_group_id", "string"),
        ColumnSpec("view_id", "string"),
        ColumnSpec("frame_index", "int"),
        ColumnSpec("time_sec", "float"),
        ColumnSpec("homography_matrix", "json"),
        ColumnSpec("source_points", "json"),
        ColumnSpec("target_points", "json"),
        ColumnSpec("confidence", "float"),
        ColumnSpec("method", "string"),
        ColumnSpec("valid", "bool"),
    ),
)


RAW_CV_SCHEMAS = {
    schema.name: schema
    for schema in (
        DETECTIONS_SCHEMA,
        TRACKS_SCHEMA,
        POSE2D_SCHEMA,
        OBJECTS_SCHEMA,
        BAT_LINES_SCHEMA,
        HOMOGRAPHY_SCHEMA,
    )
}
