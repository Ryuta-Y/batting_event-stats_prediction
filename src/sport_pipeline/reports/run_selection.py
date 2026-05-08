"""Helpers for choosing a reportable prediction run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPORT_RUN_ID_KEYS = (
    "fusion_run_id",
    "sequence_tcn_run_id",
    "video_frozen_run_id",
    "video_run_id",
    "video_finetune_run_id",
    "video_lightweight_run_id",
    "player_season_run_id",
    "vlm_run_id",
    "sequence_run_id",
    "recommended_context_run_id",
    "context_run_id",
)

PREDICTION_TABLE_SUFFIXES = (".parquet", ".jsonl", ".json", ".csv")


@dataclass(frozen=True)
class ReportRunSelection:
    run_id: str
    source: str
    available: bool
    checked_run_ids: tuple[str, ...]
    missing_required: tuple[str, ...]


def _unique(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return tuple(output)


def report_run_candidates(run_profile: dict[str, Any], *, include_smoke: bool = False) -> tuple[str, ...]:
    """Return preferred report run ids from strongest to simplest baseline."""

    run_ids = run_profile.get("run_ids", {})
    values = [str(run_ids.get(key) or "") for key in REPORT_RUN_ID_KEYS]
    if include_smoke:
        values.append("context_constant_mean_smoke")
    return _unique(values)


def prediction_artifact_path(base_dir: str | Path, run_id: str) -> Path:
    """Return the existing predictions_v1 table path, accepting all table formats."""

    base = Path(base_dir)
    run_dir = base / "predictions" / run_id
    for suffix in PREDICTION_TABLE_SUFFIXES:
        path = run_dir / f"predictions_v1{suffix}"
        if path.exists():
            return path
    return run_dir / "predictions_v1.parquet"


def metrics_artifact_path(base_dir: str | Path, run_id: str) -> Path:
    return Path(base_dir) / "predictions" / run_id / "metrics_v1.json"


def required_report_inputs(base_dir: str | Path, run_id: str) -> tuple[Path, Path]:
    return (prediction_artifact_path(base_dir, run_id), metrics_artifact_path(base_dir, run_id))


def report_inputs_exist(base_dir: str | Path, run_id: str) -> bool:
    return all(path.exists() for path in required_report_inputs(base_dir, run_id))


def select_report_run_id(
    base_dir: str | Path,
    run_profile: dict[str, Any],
    *,
    preferred_run_id: str | None = None,
    include_smoke: bool = False,
) -> ReportRunSelection:
    """Select the best available prediction run for static reports.

    Fusion remains the preferred final report target, but early Colab smoke
    runs should still be able to open reports immediately after the context
    baseline is produced.
    """

    candidates = list(report_run_candidates(run_profile, include_smoke=include_smoke))
    if preferred_run_id:
        candidates.insert(0, preferred_run_id)
    checked = _unique(candidates)
    for run_id in checked:
        if report_inputs_exist(base_dir, run_id):
            source = "preferred" if preferred_run_id and run_id == preferred_run_id else "auto_existing_artifact"
            return ReportRunSelection(run_id, source, True, checked, ())
    fallback = checked[0] if checked else str(preferred_run_id or "fusion_mlb_2024_2026_v2")
    missing = tuple(str(path) for path in required_report_inputs(base_dir, fallback) if not path.exists())
    return ReportRunSelection(fallback, "missing_fallback", False, checked, missing)
