"""Diagnostics for CV clip lifecycle outputs."""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any, Iterable


def is_clean_trainable_clip(row: dict[str, Any]) -> bool:
    """Return true when a clips_v1 row is valid for the clean training cohort."""

    return (
        row.get("clip_status") == "clean_clip"
        and row.get("quality_tier") == "usable_primary"
        and bool(row.get("clean_cohort_eligible"))
        and row.get("target_alignment_status") == "event_aligned"
    )


def _as_flag_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _top_counts(rows: Iterable[dict[str, Any]], field: str, limit: int) -> dict[str, int]:
    counter = Counter(str(row.get(field) or "missing") for row in rows)
    return dict(counter.most_common(limit))


def _top_flag_counts(rows: Iterable[dict[str, Any]], field: str, limit: int) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        values = _as_flag_items(row.get(field))
        if not values:
            counter["none"] += 1
        else:
            counter.update(values)
    return dict(counter.most_common(limit))


def _numeric_summary(rows: Iterable[dict[str, Any]], field: str) -> dict[str, float | int | None]:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if value is None or isinstance(value, bool):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    values.sort()
    return {
        "count": len(values),
        "min": values[0],
        "median": median(values),
        "max": values[-1],
    }


def _sample_rows(rows: Iterable[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    fields = (
        "clip_id",
        "event_id",
        "video_source_id",
        "clip_status",
        "quality_tier",
        "review_reason",
        "exclusion_reason",
        "target_alignment_status",
        "view_label",
        "view_confidence",
        "contact_confidence",
        "batter_visible",
        "bat_visible",
        "plate_visible",
        "quality_flags",
        "outlier_flags",
        "clip_path",
        "debug_frame_path",
    )
    sampled = []
    for row in rows:
        sampled.append({field: row.get(field) for field in fields if field in row})
        if len(sampled) >= limit:
            break
    return sampled


def clip_quality_diagnostics(rows: Iterable[dict[str, Any]], *, limit: int = 12) -> dict[str, Any]:
    """Summarize why clips_v1 rows are or are not trainable."""

    row_list = list(rows)
    clean_rows = [row for row in row_list if is_clean_trainable_clip(row)]
    non_clean_rows = [row for row in row_list if not is_clean_trainable_clip(row)]
    return {
        "total_clips": len(row_list),
        "clean_trainable_clips": len(clean_rows),
        "with_clip_path": sum(1 for row in row_list if row.get("clip_path")),
        "clip_status_counts": _top_counts(row_list, "clip_status", limit),
        "quality_tier_counts": _top_counts(row_list, "quality_tier", limit),
        "review_reason_counts": _top_counts(row_list, "review_reason", limit),
        "exclusion_reason_counts": _top_counts(row_list, "exclusion_reason", limit),
        "target_alignment_counts": _top_counts(row_list, "target_alignment_status", limit),
        "view_label_counts": _top_counts(row_list, "view_label", limit),
        "quality_flag_counts": _top_flag_counts(row_list, "quality_flags", limit),
        "outlier_flag_counts": _top_flag_counts(row_list, "outlier_flags", limit),
        "view_confidence": _numeric_summary(row_list, "view_confidence"),
        "contact_confidence": _numeric_summary(row_list, "contact_confidence"),
        "sample_clean_rows": _sample_rows(clean_rows, min(limit, 5)),
        "sample_non_clean_rows": _sample_rows(non_clean_rows, min(limit, 10)),
    }


def format_clip_quality_diagnostics(diagnostics: dict[str, Any]) -> str:
    """Render clip diagnostics as a compact Colab-friendly text block."""

    sections = [
        ("total_clips", diagnostics.get("total_clips")),
        ("clean_trainable_clips", diagnostics.get("clean_trainable_clips")),
        ("with_clip_path", diagnostics.get("with_clip_path")),
        ("clip_status_counts", diagnostics.get("clip_status_counts")),
        ("quality_tier_counts", diagnostics.get("quality_tier_counts")),
        ("review_reason_counts", diagnostics.get("review_reason_counts")),
        ("target_alignment_counts", diagnostics.get("target_alignment_counts")),
        ("view_label_counts", diagnostics.get("view_label_counts")),
        ("quality_flag_counts", diagnostics.get("quality_flag_counts")),
        ("outlier_flag_counts", diagnostics.get("outlier_flag_counts")),
        ("view_confidence", diagnostics.get("view_confidence")),
        ("contact_confidence", diagnostics.get("contact_confidence")),
        ("sample_non_clean_rows", diagnostics.get("sample_non_clean_rows")),
        ("sample_clean_rows", diagnostics.get("sample_clean_rows")),
    ]
    lines = ["clip quality diagnostics"]
    lines.extend(f"- {name}: {value}" for name, value in sections)
    return "\n".join(lines)
