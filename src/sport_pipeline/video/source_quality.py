"""Video-source quality gates for play-level batting clips."""

from __future__ import annotations

from typing import Any


EVENT_LEVEL_EVIDENCE_MARKERS = (
    "player_name_text_match",
    "event_text_match",
    "description_text_match",
    "exact_play_id",
)

MANUAL_SOURCE_KINDS = {
    "local_file",
    "manual_upload",
    "research_local",
    "user_labeled_local",
}


def has_event_level_match_reason(reason: str | None) -> bool:
    """Return true when a source match reason points to the play, not only the game."""

    text = str(reason or "")
    return any(marker in text for marker in EVENT_LEVEL_EVIDENCE_MARKERS)


def is_exact_play_savant_source(row: dict[str, Any]) -> bool:
    """Return true for direct Baseball Savant play-page media."""

    return (
        str(row.get("source_kind") or "") == "baseball_savant_sporty"
        and bool(row.get("media_url"))
        and "exact_play_id" in str(row.get("match_reason") or "")
    )


def _float_or_default(value: Any, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_manual_research_source(row: dict[str, Any]) -> bool:
    """Allow explicitly labeled local/manual videos without public URL evidence."""

    source_kind = str(row.get("source_kind") or "")
    if source_kind in MANUAL_SOURCE_KINDS:
        return True
    local_path = row.get("local_video_path")
    review_status = str(row.get("review_status") or "")
    return bool(local_path) and review_status in {"usable_primary", "clean_clip", "approved"}


def is_event_level_video_source(
    row: dict[str, Any],
    *,
    min_match_confidence: float = 0.45,
) -> bool:
    """Return true when a video source is specific enough for training download/CV."""

    if is_manual_research_source(row):
        return True
    if is_exact_play_savant_source(row):
        return True
    if not bool(row.get("media_url")) and not bool(row.get("local_video_path")):
        return False
    if not has_event_level_match_reason(str(row.get("match_reason") or "")):
        return False
    return _float_or_default(row.get("match_confidence"), 0.0) >= min_match_confidence


def video_source_skip_reason(
    row: dict[str, Any],
    *,
    min_match_confidence: float = 0.45,
) -> str | None:
    """Return a reason when a row should not feed training video download/CV."""

    if is_event_level_video_source(row, min_match_confidence=min_match_confidence):
        return None
    if not row.get("media_url") and not row.get("local_video_path"):
        return "missing_media_url_or_local_video"
    if not has_event_level_match_reason(str(row.get("match_reason") or "")):
        return "not_event_level_match"
    confidence = _float_or_default(row.get("match_confidence"), 0.0)
    if confidence < min_match_confidence:
        return f"match_confidence_below_{min_match_confidence:g}"
    return "not_event_level_video_source"


def source_priority(row: dict[str, Any]) -> tuple[int, float, str]:
    """Sort key that puts exact play clips before broad game-content videos."""

    if is_exact_play_savant_source(row):
        tier = 0
    elif is_manual_research_source(row):
        tier = 1
    elif is_event_level_video_source(row):
        tier = 2
    else:
        tier = 9
    confidence = _float_or_default(row.get("match_confidence"), 0.0)
    return (tier, -confidence, str(row.get("video_source_id") or ""))
