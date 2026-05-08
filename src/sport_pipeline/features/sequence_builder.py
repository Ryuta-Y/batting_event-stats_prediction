"""Lightweight structured sequence metadata builders."""

from __future__ import annotations

from typing import Any

from sport_pipeline.features.sequence_contracts import SEQUENCE_CONTRACT_VERSION


DEFAULT_FEATURE_NAMES = (
    "pelvis_x",
    "pelvis_y",
    "trunk_angle",
    "lead_hand_speed",
    "rear_hand_speed",
    "bat_angle",
    "bat_angular_velocity",
    "bat_visible_length",
    "plate_x_canonical",
    "plate_z_canonical",
)

PHASE_LABELS = (
    "stance_load",
    "stride",
    "launch",
    "swing",
    "contact",
    "follow_through",
    "unknown",
)


def build_sequence_id(clip_id: str, feature_namespace: str = "structured_batting_kinematics_v1") -> str:
    """Build a stable sequence id from one clip."""

    return f"{clip_id}__{feature_namespace}"


def build_sequence_manifest_row(
    clip_row: dict[str, Any],
    n_frames: int,
    feature_names: list[str] | tuple[str, ...] = DEFAULT_FEATURE_NAMES,
    split: str = "unknown",
    sequence_path: str | None = None,
    target_available: bool = True,
    target_missing_reason: str | None = None,
) -> dict[str, Any]:
    """Build one structured_sequence_v1 manifest row from clips_v1 metadata."""

    sequence_id = build_sequence_id(str(clip_row["clip_id"]))
    return {
        "schema_version": SEQUENCE_CONTRACT_VERSION,
        "sequence_id": sequence_id,
        "sample_id": sequence_id,
        "clip_id": clip_row["clip_id"],
        "event_id": clip_row["event_id"],
        "same_event_group_id": clip_row["same_event_group_id"],
        "view_id": clip_row["view_id"],
        "batter_id": clip_row["batter_id"],
        "season": clip_row["season"],
        "batter_season_id": clip_row["batter_season_id"],
        "game_date": clip_row.get("game_date", "1970-01-01"),
        "split": split,
        "sequence_path": sequence_path,
        "n_frames": n_frames,
        "feature_dim": len(feature_names),
        "feature_names": list(feature_names),
        "phase_labels": list(PHASE_LABELS),
        "frame_rate": clip_row["fps"],
        "clip_version": clip_row["clip_version"],
        "clip_status": clip_row["clip_status"],
        "quality_tier": clip_row["quality_tier"],
        "view_label": clip_row["view_label"],
        "target_available": target_available,
        "target_missing_reason": target_missing_reason,
    }


def build_frame_feature_row(
    sequence_manifest_row: dict[str, Any],
    frame_index: int,
    time_sec: float,
    feature_values: list[float],
    relative_time_to_contact_sec: float | None = None,
    phase_label: str = "unknown",
    phase_confidence: float = 0.0,
    feature_mask: list[bool] | None = None,
    feature_namespace: str = "structured_batting_kinematics_v1",
) -> dict[str, Any]:
    """Build one frame-level T x D feature row."""

    feature_names = list(sequence_manifest_row["feature_names"])
    if len(feature_values) != len(feature_names):
        raise ValueError("feature_values length must match feature_names")
    if feature_mask is None:
        feature_mask = [True] * len(feature_names)
    if len(feature_mask) != len(feature_names):
        raise ValueError("feature_mask length must match feature_names")
    return {
        "schema_version": SEQUENCE_CONTRACT_VERSION,
        "sequence_id": sequence_manifest_row["sequence_id"],
        "clip_id": sequence_manifest_row["clip_id"],
        "event_id": sequence_manifest_row["event_id"],
        "same_event_group_id": sequence_manifest_row["same_event_group_id"],
        "view_id": sequence_manifest_row["view_id"],
        "batter_id": sequence_manifest_row["batter_id"],
        "season": sequence_manifest_row["season"],
        "batter_season_id": sequence_manifest_row["batter_season_id"],
        "frame_index": frame_index,
        "time_sec": time_sec,
        "relative_time_to_contact_sec": relative_time_to_contact_sec,
        "phase_label": phase_label,
        "phase_confidence": phase_confidence,
        "feature_namespace": feature_namespace,
        "feature_names": feature_names,
        "feature_values": feature_values,
        "feature_mask": feature_mask,
        "view_label": sequence_manifest_row["view_label"],
        "quality_tier": sequence_manifest_row["quality_tier"],
        "clip_status": sequence_manifest_row["clip_status"],
    }

