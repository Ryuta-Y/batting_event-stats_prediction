"""Report summaries for predictions, metrics, clip metadata, and fusion audit rows."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import isnan
from typing import Any, Iterable, Mapping

from sport_pipeline.evaluation.predictions import validate_prediction_rows


OPTIONAL_EVENT_TARGETS = ("xba", "xwoba")
OPS_TARGET = "ops"
CLIP_STATUSES = ("clean_clip", "review_only", "excluded")


def _reason(value: object | None, fallback: str = "missing_reason_unknown") -> str:
    if value in (None, ""):
        return fallback
    return str(value)


def _float_or_none(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    if isnan(output):
        return None
    return output


def _join_reason_counts(counts: Mapping[str, int]) -> str:
    if not counts:
        return ""
    return ", ".join(f"{reason}:{count}" for reason, count in sorted(counts.items()))


def target_availability_summary(prediction_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count available and missing labels per target from predictions_v1 rows."""

    rows = list(prediction_rows)
    validate_prediction_rows(rows)
    summary: dict[str, dict[str, Any]] = {}
    for target in sorted({row["target_name"] for row in rows}.union(OPTIONAL_EVENT_TARGETS, {OPS_TARGET})):
        summary[target] = {
            "target_name": target,
            "prediction_level": "",
            "available": 0,
            "missing": 0,
            "missing_reasons": Counter(),
        }
    for row in rows:
        target = str(row["target_name"])
        item = summary[target]
        if not item["prediction_level"]:
            item["prediction_level"] = str(row["prediction_level"])
        if bool(row["target_available"]) and row.get("y_true") is not None:
            item["available"] += 1
        else:
            item["missing"] += 1
            item["missing_reasons"][_reason(row.get("label_missing_reason"))] += 1

    output = []
    for item in summary.values():
        output.append(
            {
                "target_name": item["target_name"],
                "prediction_level": item["prediction_level"] or ("player_season" if item["target_name"] == OPS_TARGET else "event"),
                "available": item["available"],
                "missing": item["missing"],
                "missing_reasons": _join_reason_counts(item["missing_reasons"]),
            }
        )
    return sorted(output, key=lambda row: (row["prediction_level"], row["target_name"]))


def ops_unavailable_reasons(prediction_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return explicit OPS unavailable reasons."""

    rows = list(prediction_rows)
    validate_prediction_rows(rows)
    reasons = Counter(
        _reason(row.get("label_missing_reason"), "ops_unavailable")
        for row in rows
        if row.get("target_name") == OPS_TARGET and not row.get("target_available")
    )
    if not reasons and not any(row.get("target_name") == OPS_TARGET for row in rows):
        reasons["pa_manifest_unavailable"] = 0
    return [{"reason": reason, "count": count} for reason, count in sorted(reasons.items())]


def clip_quality_counts(clip_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count clean/review/excluded clip lifecycle statuses."""

    counts = Counter()
    reasons: dict[str, Counter] = defaultdict(Counter)
    tiers: dict[str, Counter] = defaultdict(Counter)
    for row in clip_rows:
        status = str(row.get("clip_status") or row.get("review_status") or "unknown")
        counts[status] += 1
        reasons[status][_reason(row.get("review_reason") or row.get("exclusion_reason") or row.get("status_reason"), "none")] += 1
        tiers[status][_reason(row.get("quality_tier"), "unknown")] += 1
    for status in CLIP_STATUSES:
        counts.setdefault(status, 0)
    return [
        {
            "clip_status": status,
            "count": count,
            "quality_tiers": _join_reason_counts(tiers[status]),
            "reasons": _join_reason_counts(reasons[status]),
        }
        for status, count in sorted(counts.items())
    ]


def ensemble_prior_summary(
    prediction_rows: Iterable[dict[str, Any]],
    fusion_audit_rows: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Summarize same-event ensemble and player-season prior metadata."""

    rows = list(prediction_rows)
    validate_prediction_rows(rows)
    summary = [
        {
            "comparison_axis": "same-event ensemble",
            "rows": sum(1 for row in rows if bool(row.get("same_event_ensemble"))),
            "available_rows": sum(1 for row in rows if bool(row.get("same_event_ensemble")) and bool(row.get("target_available"))),
            "scopes": _join_reason_counts(Counter(str(row.get("aggregation_scope")) for row in rows if bool(row.get("same_event_ensemble")))),
        },
        {
            "comparison_axis": "player-season prior",
            "rows": sum(1 for row in rows if int(row.get("n_prior_clips") or 0) > 0 or str(row.get("prior_mode")) != "none"),
            "available_rows": sum(
                1
                for row in rows
                if (int(row.get("n_prior_clips") or 0) > 0 or str(row.get("prior_mode")) != "none")
                and bool(row.get("target_available"))
            ),
            "scopes": _join_reason_counts(Counter(str(row.get("prior_mode")) for row in rows if int(row.get("n_prior_clips") or 0) > 0 or str(row.get("prior_mode")) != "none")),
        },
    ]
    if fusion_audit_rows is not None:
        audit_rows = list(fusion_audit_rows)
        summary.append(
            {
                "comparison_axis": "fusion input audit",
                "rows": len(audit_rows),
                "available_rows": sum(1 for row in audit_rows if bool(row.get("source_target_available"))),
                "scopes": _join_reason_counts(Counter(str(row.get("source_aggregation_scope")) for row in audit_rows)),
            }
        )
    return summary


def experiment_compare_summary(metrics_payloads: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten metrics_v1 payloads into a model comparison table."""

    output: list[dict[str, Any]] = []
    for payload in metrics_payloads:
        run_id = str(payload.get("run_id", "unknown_run"))
        metrics = payload.get("metrics", {})
        skipped = payload.get("skipped", {})
        for prediction_level, target_metrics in metrics.items():
            for target_name, values in target_metrics.items():
                metric_values = {
                    key: value
                    for key, value in values.items()
                    if key not in {"n_available", "n_skipped"} and value is not None
                }
                output.append(
                    {
                        "run_id": run_id,
                        "prediction_level": prediction_level,
                        "target_name": target_name,
                        "n_available": values.get("n_available", 0),
                        "n_skipped": values.get("n_skipped", 0),
                        "metrics": ", ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}" for key, value in sorted(metric_values.items())),
                        "skip_reasons": _join_reason_counts(skipped.get(target_name, {})),
                    }
                )
    return sorted(output, key=lambda row: (row["prediction_level"], row["target_name"], row["run_id"]))


def failure_cases(
    prediction_rows: Iterable[dict[str, Any]],
    clip_rows: Iterable[dict[str, Any]] | None = None,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Select static failure cards sorted by absolute error, then missing labels."""

    rows = list(prediction_rows)
    validate_prediction_rows(rows)
    clip_by_event: dict[str, dict[str, Any]] = {}
    if clip_rows is not None:
        for clip in clip_rows:
            event_id = clip.get("event_id")
            if event_id is not None and str(event_id) not in clip_by_event:
                clip_by_event[str(event_id)] = clip

    cards: list[dict[str, Any]] = []
    for row in rows:
        y_true = _float_or_none(row.get("y_true"))
        y_pred = _float_or_none(row.get("y_pred"))
        error = None if y_true is None or y_pred is None else y_pred - y_true
        abs_error = None if error is None else abs(error)
        if abs_error is None and row.get("target_available"):
            continue
        event_id = row.get("event_id")
        clip = clip_by_event.get(str(event_id)) if event_id is not None else None
        links = []
        if clip is not None:
            for label, key in (("clip", "clip_path"), ("overlay", "overlay_path"), ("debug frame", "debug_frame_path")):
                path = clip.get(key)
                if path:
                    links.append((label, str(path)))
        cards.append(
            {
                "title": f"{row['target_name']} {row['sample_id']}",
                "run_id": row["run_id"],
                "sample_id": row["sample_id"],
                "event_id": row.get("event_id"),
                "batter_season_id": row["batter_season_id"],
                "prediction_level": row["prediction_level"],
                "target_name": row["target_name"],
                "y_true": row.get("y_true"),
                "y_pred": row.get("y_pred"),
                "error": None if error is None else round(error, 6),
                "abs_error": None if abs_error is None else round(abs_error, 6),
                "target_available": row["target_available"],
                "label_missing_reason": row.get("label_missing_reason"),
                "aggregation_scope": row["aggregation_scope"],
                "prior_mode": row["prior_mode"],
                "same_event_ensemble": row.get("same_event_ensemble"),
                "n_prior_clips": row.get("n_prior_clips"),
                "clip_status": clip.get("clip_status") if clip else None,
                "quality_tier": clip.get("quality_tier") if clip else None,
                "view_label": clip.get("view_label") if clip else None,
                "links": links,
            }
        )
    cards.sort(key=lambda card: (card["abs_error"] is None, -(card["abs_error"] or 0.0), str(card["target_name"])))
    return cards[:limit]
