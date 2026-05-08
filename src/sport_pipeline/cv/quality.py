"""Quality and lifecycle helpers for CV preprocessing metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sport_pipeline.cv.contracts import CV_CONTRACT_VERSION


CLEAN_CLIP_STATUS = "clean_clip"
REVIEW_ONLY_STATUS = "review_only"
EXCLUDED_STATUS = "excluded"

USABLE_PRIMARY_TIER = "usable_primary"
USABLE_WITH_OUTLIERS_TIER = "usable_with_outliers"
REVIEW_ONLY_TIER = "review_only"
EXCLUDED_TIER = "excluded"

BATTER_MECHANICS_VIEWS = (
    "bat_catcher_view",
    "pitch_catcher_view",
    "broadcast_infield",
    "bat_side",
    "bat_pitcher_view",
)

NON_BATTING_VIEWS = ("runner_only", "dugout", "crowd", "graphic", "ball_flight")
REPLAY_VIEWS = ("replay_closeup",)
SEVERE_OUTLIER_FLAGS = (
    "pose_extreme_posture",
    "post_contact_fallaway",
    "lost_balance",
    "check_swing_like",
    "contact_not_visible",
    "shot_cut_near_contact",
)

TRIM_POLICIES = {
    "pre_contact_long": (-2.00, 1.00),
    "contact_window": (-0.50, 0.30),
    "full_swing": (-1.60, 0.60),
    "raw_source_window": (0.0, 0.0),
}


@dataclass(frozen=True)
class CVQualityPolicy:
    min_view_confidence: float = 0.50
    min_contact_confidence: float = 0.60
    min_batter_visibility: float = 0.65
    min_bat_visibility: float = 0.45
    min_plate_visibility: float = 0.30
    clean_zones: tuple[int, ...] = (2, 4, 5, 6, 8)
    max_clean_strikes: int = 1
    clean_views: tuple[str, ...] = BATTER_MECHANICS_VIEWS


DEFAULT_CV_QUALITY_POLICY = CVQualityPolicy()


def normalize_flags(value: Any) -> tuple[str, ...]:
    """Normalize flags stored as list/tuple/set/comma-separated string."""

    if value is None:
        return ()
    if isinstance(value, str):
        if not value.strip():
            return ()
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value if str(item))
    return (str(value),)


def detect_difficult_pitch(row: dict[str, Any], policy: CVQualityPolicy = DEFAULT_CV_QUALITY_POLICY) -> bool:
    """Return true when pitch context should stay out of the initial clean cohort."""

    if bool(row.get("difficult_pitch")):
        return True
    zone = row.get("zone")
    if zone is not None and int(zone) not in set(policy.clean_zones):
        return True
    strikes = row.get("strikes")
    if strikes is not None and int(strikes) > policy.max_clean_strikes:
        return True
    return False


def visibility_quality_flags(
    row: dict[str, Any],
    policy: CVQualityPolicy = DEFAULT_CV_QUALITY_POLICY,
) -> tuple[str, ...]:
    """Build quality flags for batter, bat, plate, contact, view, and shot issues."""

    flags: list[str] = []
    if not bool(row.get("batter_visible")) or float(row.get("batter_visibility_score", 0.0)) < policy.min_batter_visibility:
        flags.append("batter_low_visibility")
    if not bool(row.get("bat_visible")) or float(row.get("bat_visibility_score", 0.0)) < policy.min_bat_visibility:
        flags.append("bat_low_visibility")
    if not bool(row.get("plate_visible")) or float(row.get("plate_visibility_score", 0.0)) < policy.min_plate_visibility:
        flags.append("plate_low_visibility")
    if not bool(row.get("contact_visible")) or float(row.get("contact_confidence", 0.0)) < policy.min_contact_confidence:
        flags.append("contact_not_visible")
    if float(row.get("view_confidence", 0.0)) < policy.min_view_confidence:
        flags.append("view_low_confidence")
    if bool(row.get("is_broadcast_cutaway")):
        flags.append("broadcast_cutaway")
    if bool(row.get("is_dugout")):
        flags.append("dugout")
    if bool(row.get("is_non_batting_segment")):
        flags.append("non_batting_segment")
    if bool(row.get("is_replay")):
        flags.append("replay")
    return tuple(dict.fromkeys(flags))


def outlier_flags_for_row(row: dict[str, Any]) -> tuple[str, ...]:
    """Combine existing and derived outlier flags without dropping hard cases."""

    flags = list(normalize_flags(row.get("outlier_flags")))
    if bool(row.get("extreme_form")):
        flags.append("pose_extreme_posture")
    if bool(row.get("lost_balance")):
        flags.append("lost_balance")
    if bool(row.get("check_swing_like")):
        flags.append("check_swing_like")
    if bool(row.get("shot_cut_near_contact")):
        flags.append("shot_cut_near_contact")
    return tuple(dict.fromkeys(flags))


def classify_candidate_segment(
    row: dict[str, Any],
    policy: CVQualityPolicy = DEFAULT_CV_QUALITY_POLICY,
) -> dict[str, Any]:
    """Classify a candidate segment into clean, review-only, or excluded."""

    view_label = str(row.get("view_label", "unknown"))
    quality_flags = set(visibility_quality_flags(row, policy))
    outlier_flags = set(outlier_flags_for_row(row))
    difficult_pitch = detect_difficult_pitch(row, policy)
    extreme_form = bool(row.get("extreme_form")) or bool(outlier_flags.intersection(SEVERE_OUTLIER_FLAGS))
    non_batting = bool(row.get("is_non_batting_segment")) or view_label in NON_BATTING_VIEWS
    replay_or_cutaway = bool(row.get("is_replay")) or bool(row.get("is_broadcast_cutaway")) or view_label in REPLAY_VIEWS
    hard_negative = non_batting or replay_or_cutaway

    if bool(row.get("wrong_event")):
        return {
            "clip_status": EXCLUDED_STATUS,
            "quality_tier": EXCLUDED_TIER,
            "review_status": EXCLUDED_STATUS,
            "status_reason": "wrong_event",
            "clean_cohort_eligible": False,
            "robustness_cohort_eligible": False,
            "hard_negative": hard_negative,
            "difficult_pitch": difficult_pitch,
            "extreme_form": extreme_form,
            "quality_flags": tuple(sorted(quality_flags)),
            "outlier_flags": tuple(sorted(outlier_flags)),
            "target_alignment_status": "wrong_event",
        }

    if non_batting:
        reason = "non_batting_segment"
    elif replay_or_cutaway:
        reason = "replay_or_cutaway"
    elif view_label not in policy.clean_views:
        reason = "unsupported_view"
    elif quality_flags:
        reason = ",".join(sorted(quality_flags))
    elif difficult_pitch:
        reason = "difficult_pitch"
    elif extreme_form:
        reason = "extreme_form"
    else:
        reason = None

    if reason is None:
        return {
            "clip_status": CLEAN_CLIP_STATUS,
            "quality_tier": USABLE_PRIMARY_TIER,
            "review_status": CLEAN_CLIP_STATUS,
            "status_reason": None,
            "clean_cohort_eligible": True,
            "robustness_cohort_eligible": True,
            "hard_negative": False,
            "difficult_pitch": False,
            "extreme_form": False,
            "quality_flags": tuple(sorted(quality_flags)),
            "outlier_flags": tuple(sorted(outlier_flags)),
            "target_alignment_status": "event_aligned",
        }

    quality_tier = REVIEW_ONLY_TIER if hard_negative or "contact_not_visible" in quality_flags else USABLE_WITH_OUTLIERS_TIER
    return {
        "clip_status": REVIEW_ONLY_STATUS,
        "quality_tier": quality_tier,
        "review_status": REVIEW_ONLY_STATUS,
        "status_reason": reason,
        "clean_cohort_eligible": False,
        "robustness_cohort_eligible": not non_batting,
        "hard_negative": hard_negative,
        "difficult_pitch": difficult_pitch,
        "extreme_form": extreme_form,
        "quality_flags": tuple(sorted(quality_flags)),
        "outlier_flags": tuple(sorted(outlier_flags)),
        "target_alignment_status": "needs_review",
    }


def trim_window(
    contact_time_sec: float | None,
    segment_start_sec: float,
    segment_end_sec: float,
    clip_version: str = "pre_contact_long",
) -> tuple[float | None, float | None, float, float]:
    """Return contact-centered trim bounds clipped to the segment extent."""

    pre_contact, post_contact = TRIM_POLICIES[clip_version]
    if contact_time_sec is None:
        return None, None, abs(pre_contact), post_contact
    if clip_version == "raw_source_window":
        return segment_start_sec, segment_end_sec, 0.0, 0.0
    start = max(segment_start_sec, contact_time_sec + pre_contact)
    end = min(segment_end_sec, contact_time_sec + post_contact)
    return start, end, abs(pre_contact), post_contact


def build_clip_id(
    event_id: str,
    view_id: str,
    clip_version: str,
    crop_id: str = "full_frame",
    augmentation_id: str = "orig",
) -> str:
    """Build a stable clip id while keeping same-event grouping separate."""

    return f"{event_id}__{view_id}__{clip_version}__{crop_id}__{augmentation_id}"


def derive_clip_metadata(
    candidate_segment: dict[str, Any],
    clip_version: str = "pre_contact_long",
    crop_id: str = "full_frame",
    augmentation_id: str = "orig",
    policy: CVQualityPolicy = DEFAULT_CV_QUALITY_POLICY,
) -> dict[str, Any]:
    """Derive a clips_v1 metadata row from a candidate segment row."""

    decision = classify_candidate_segment(candidate_segment, policy)
    trim_start, trim_end, pre_contact, post_contact = trim_window(
        candidate_segment.get("contact_time_sec"),
        float(candidate_segment["start_time_sec"]),
        float(candidate_segment["end_time_sec"]),
        clip_version=clip_version,
    )
    event_id = str(candidate_segment["event_id"])
    view_id = str(candidate_segment["view_id"])
    clip_status = decision["clip_status"]
    return {
        "schema_version": CV_CONTRACT_VERSION,
        "clip_id": build_clip_id(event_id, view_id, clip_version, crop_id, augmentation_id),
        "candidate_segment_id": candidate_segment["candidate_segment_id"],
        "video_source_id": candidate_segment["video_source_id"],
        "event_id": event_id,
        "same_event_group_id": candidate_segment["same_event_group_id"],
        "view_id": view_id,
        "crop_id": crop_id,
        "augmentation_id": augmentation_id,
        "game_pk": candidate_segment["game_pk"],
        "play_id": candidate_segment.get("play_id"),
        "batter_id": candidate_segment["batter_id"],
        "season": candidate_segment["season"],
        "batter_season_id": candidate_segment["batter_season_id"],
        "clip_version": clip_version,
        "clip_status": clip_status,
        "quality_tier": decision["quality_tier"],
        "dataset_role": candidate_segment["dataset_role"],
        "cohort_role": "clean_location_cohort_v1" if decision["clean_cohort_eligible"] else "robustness_or_review",
        "clip_path": None,
        "overlay_path": None,
        "debug_frame_path": None,
        "start_frame": candidate_segment["start_frame"],
        "end_frame": candidate_segment["end_frame"],
        "start_time_sec": trim_start if trim_start is not None else candidate_segment["start_time_sec"],
        "end_time_sec": trim_end if trim_end is not None else candidate_segment["end_time_sec"],
        "fps": candidate_segment["fps"],
        "duration_sec": (trim_end - trim_start) if trim_start is not None and trim_end is not None else candidate_segment["duration_sec"],
        "contact_frame": candidate_segment.get("contact_frame"),
        "contact_time_sec": candidate_segment.get("contact_time_sec"),
        "contact_confidence": candidate_segment["contact_confidence"],
        "contact_window_policy": clip_version,
        "pre_contact_sec": pre_contact,
        "post_contact_sec": post_contact,
        "view_label": candidate_segment["view_label"],
        "view_confidence": candidate_segment["view_confidence"],
        "batter_visible": candidate_segment["batter_visible"],
        "batter_visibility_score": candidate_segment["batter_visibility_score"],
        "bat_visible": candidate_segment["bat_visible"],
        "bat_visibility_score": candidate_segment["bat_visibility_score"],
        "plate_visible": candidate_segment["plate_visible"],
        "plate_visibility_score": candidate_segment["plate_visibility_score"],
        "clean_cohort_eligible": decision["clean_cohort_eligible"],
        "robustness_cohort_eligible": decision["robustness_cohort_eligible"],
        "hard_negative": decision["hard_negative"],
        "difficult_pitch": decision["difficult_pitch"],
        "extreme_form": decision["extreme_form"],
        "review_reason": decision["status_reason"] if clip_status == REVIEW_ONLY_STATUS else None,
        "exclusion_reason": decision["status_reason"] if clip_status == EXCLUDED_STATUS else None,
        "target_alignment_status": decision["target_alignment_status"],
        "join_key_fields": ["event_id", "game_pk", "play_id", "batter_id", "season", "same_event_group_id"],
        "quality_flags": list(decision["quality_flags"]),
        "outlier_flags": list(decision["outlier_flags"]),
    }
