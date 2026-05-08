"""Research-facing pipeline dashboard for full Colab batting runs."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from sport_pipeline.io import read_table
from sport_pipeline.pipeline.run_profile import resolve_statcast_date_range
from sport_pipeline.reports.build_static import read_metrics_payload
from sport_pipeline.reports.html import render_kv_table, render_page, render_table, write_page
from sport_pipeline.reports.run_selection import (
    metrics_artifact_path,
    prediction_artifact_path,
    report_run_candidates,
)
from sport_pipeline.reports.summaries import experiment_compare_summary


TABLE_SUFFIXES = (".parquet", ".jsonl", ".json", ".csv")


def _format_path_template(template: str, run_profile: dict[str, Any], *, today: date | None = None) -> str:
    run_ids = {key: str(value) for key, value in run_profile.get("run_ids", {}).items()}
    start_date, end_date = resolve_statcast_date_range(run_profile, today=today)
    values = {
        **run_ids,
        "start_date": start_date,
        "end_date": end_date,
    }
    try:
        return template.format(**values)
    except KeyError:
        return template


def _size_mb(path: Path) -> float | None:
    if not path.exists() or path.is_dir():
        return None
    return round(path.stat().st_size / 1_048_576, 3)


def _count_json_rows(path: Path) -> int | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return len(payload["rows"])
    return None


def _table_row_count(path: Path) -> int | None:
    if not path.exists() or path.is_dir():
        return None
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    if suffix == ".json":
        return _count_json_rows(path)
    if suffix == ".csv":
        with path.open("r", encoding="utf-8") as handle:
            line_count = sum(1 for _ in handle)
        return max(0, line_count - 1)
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore

            return int(pq.ParquetFile(path).metadata.num_rows)
        except Exception:
            try:
                import pandas as pd  # type: ignore

                return int(len(pd.read_parquet(path, columns=[])))
            except Exception:
                return None
    return None


def _read_json_status(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(payload, dict):
        return ""
    for key in ("status", "run_status"):
        value = payload.get(key)
        if value:
            return str(value)
    outputs = payload.get("outputs")
    if isinstance(outputs, dict):
        rendered = payload.get("rendered_overlays")
        if rendered is not None:
            return f"rendered_overlays={rendered}"
    return ""


def _artifact_kind(relative_path: str) -> str:
    if relative_path.endswith(".json"):
        return "json"
    if any(relative_path.endswith(suffix) for suffix in TABLE_SUFFIXES):
        return "table"
    if relative_path.endswith(".pt"):
        return "checkpoint"
    if relative_path.endswith(".html"):
        return "html"
    return "directory" if "." not in Path(relative_path).name else "file"


def artifact_inventory_rows(
    base_dir: str | Path,
    run_profile: dict[str, Any],
    *,
    today: date | None = None,
) -> list[dict[str, Any]]:
    """Return one row per expected artifact declared by the run profile."""

    base = Path(base_dir)
    rows: list[dict[str, Any]] = []
    for group in run_profile.get("artifact_groups", []):
        group_name = str(group.get("name", "unknown"))
        for template in group.get("artifacts", []):
            relative = _format_path_template(str(template), run_profile, today=today)
            path = base / relative
            kind = _artifact_kind(relative)
            exists = path.exists()
            row_count = _table_row_count(path) if exists and kind == "table" else None
            if exists and path.is_dir():
                row_count = sum(1 for _ in path.glob("*"))
            status = _read_json_status(path) if exists and kind == "json" else ""
            rows.append(
                {
                    "group": group_name,
                    "stage": group.get("stage", ""),
                    "required": bool(group.get("required")),
                    "artifact": relative,
                    "exists": exists,
                    "kind": kind,
                    "rows_or_files": row_count,
                    "size_mb": _size_mb(path),
                    "status": status,
                    "path": str(path),
                    "description_ja": group.get("description_ja", ""),
                }
            )
    return rows


def artifact_group_summary(inventory_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize artifact completeness by profile artifact group."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in inventory_rows:
        grouped.setdefault(str(row["group"]), []).append(row)
    output = []
    for group, rows in grouped.items():
        present = sum(1 for row in rows if row["exists"])
        total = len(rows)
        required = any(bool(row["required"]) for row in rows)
        output.append(
            {
                "group": group,
                "stage": rows[0].get("stage", ""),
                "required": required,
                "present": present,
                "total": total,
                "status": "complete" if present == total else ("missing_required" if required and present < total else "partial"),
                "description_ja": rows[0].get("description_ja", ""),
            }
        )
    return output


def reportable_run_rows(base_dir: str | Path, run_profile: dict[str, Any], *, include_smoke: bool = False) -> list[dict[str, Any]]:
    """Return prediction/metric artifact availability for each configured model run."""

    rows = []
    for run_id in report_run_candidates(run_profile, include_smoke=include_smoke):
        predictions = prediction_artifact_path(base_dir, run_id)
        metrics = metrics_artifact_path(base_dir, run_id)
        rows.append(
            {
                "run_id": run_id,
                "predictions_exists": predictions.exists(),
                "prediction_rows": _table_row_count(predictions) if predictions.exists() else None,
                "metrics_exists": metrics.exists(),
                "predictions_path": str(predictions),
                "metrics_path": str(metrics),
            }
        )
    return rows


def available_metrics_payloads(base_dir: str | Path, run_profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Read all configured metrics_v1 payloads that exist."""

    payloads = []
    for row in reportable_run_rows(base_dir, run_profile):
        metrics_path = Path(str(row["metrics_path"]))
        if row["metrics_exists"]:
            payloads.append(read_metrics_payload(metrics_path))
    return payloads


def _target_metric_winners(metric_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick the lowest MAE/Brier row per prediction level and target when visible."""

    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in metric_rows:
        metrics_text = str(row.get("metrics") or "")
        score = None
        score_name = ""
        for candidate in ("mae", "brier", "rmse"):
            marker = f"{candidate}="
            if marker in metrics_text:
                fragment = metrics_text.split(marker, 1)[1].split(",", 1)[0]
                try:
                    score = float(fragment)
                    score_name = candidate
                    break
                except ValueError:
                    continue
        if score is None:
            continue
        key = (str(row.get("prediction_level")), str(row.get("target_name")))
        current = best.get(key)
        if current is None or score < float(current["score"]):
            best[key] = {
                "prediction_level": key[0],
                "target_name": key[1],
                "best_run_id": row.get("run_id"),
                "score_name": score_name,
                "score": round(score, 6),
                "n_available": row.get("n_available"),
            }
    return sorted(best.values(), key=lambda row: (row["prediction_level"], row["target_name"]))


def visual_evidence_rows(base_dir: str | Path, full_run_id: str) -> list[dict[str, Any]]:
    """Summarize clip, CV, and overlay artifacts used for visual inspection."""

    base = Path(base_dir)
    paths = {
        "clips": base / "clips" / full_run_id / "clips_v1.parquet",
        "detections": base / "detections" / full_run_id / "detections_v1.parquet",
        "tracks": base / "tracks" / full_run_id / "tracks_v1.parquet",
        "pose skeleton": base / "pose2d" / full_run_id / "pose2d_v1.parquet",
        "bat / plate objects": base / "objects" / full_run_id / "bat_detection_v1.parquet",
        "bat lines": base / "objects" / full_run_id / "bat_line_v1.parquet",
        "homography": base / "homography" / full_run_id / "homography_v1.parquet",
        "overlay manifest": base / "debug" / full_run_id / "cv_overlay_videos_v1.parquet",
        "overlay videos": base / "debug" / full_run_id / "cv_overlays",
    }
    rows = []
    for label, path in paths.items():
        exists = path.exists()
        rows.append(
            {
                "artifact": label,
                "exists": exists,
                "rows_or_files": sum(1 for _ in path.glob("*.mp4")) if exists and path.is_dir() else _table_row_count(path),
                "path": str(path),
            }
        )
    return rows


def clip_status_rows(base_dir: str | Path, full_run_id: str) -> list[dict[str, Any]]:
    """Return compact clip lifecycle counts from the full-run clips manifest."""

    clips_path = Path(base_dir) / "clips" / full_run_id / "clips_v1.parquet"
    if not clips_path.exists():
        return []
    clips = read_table(clips_path)
    by_status = Counter(str(row.get("clip_status") or "unknown") for row in clips)
    overlay_count = sum(1 for row in clips if row.get("overlay_path"))
    output = [{"clip_status": key, "count": value} for key, value in sorted(by_status.items())]
    output.append({"clip_status": "with_overlay_path", "count": overlay_count})
    return output


def build_pipeline_dashboard(
    *,
    base_dir: str | Path,
    run_profile: dict[str, Any],
    dashboard_id: str | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Path]:
    """Build a single landing page that explains what the full run produced."""

    base = Path(base_dir)
    run_ids = run_profile.get("run_ids", {})
    full_run_id = str(run_ids.get("full_run_id", "mlb_2024_2026_full_v2"))
    final_run_id = str(run_ids.get("fusion_run_id") or run_ids.get("sequence_tcn_run_id") or full_run_id)
    report_id = dashboard_id or final_run_id
    report_root = Path(output_root) if output_root is not None else base / "reports"
    output_path = report_root / "pipeline_dashboard" / report_id / "index.html"
    summary_path = report_root / "pipeline_dashboard" / report_id / "summary.json"

    inventory = artifact_inventory_rows(base, run_profile)
    group_summary = artifact_group_summary(inventory)
    run_rows = reportable_run_rows(base, run_profile)
    metrics_payloads = available_metrics_payloads(base, run_profile)
    metric_rows = experiment_compare_summary(metrics_payloads)
    visual_rows = visual_evidence_rows(base, full_run_id)
    clip_rows = clip_status_rows(base, full_run_id)
    winner_rows = _target_metric_winners(metric_rows)

    metadata = {
        "base_dir": str(base),
        "dashboard_id": report_id,
        "full_run_id": full_run_id,
        "final_prediction_run_id": final_run_id,
        "metric_payloads_found": len(metrics_payloads),
        "artifact_groups": len(group_summary),
        "note_ja": "prediction run と clip/CV run は別IDで保存されるため、このページは run profile 全体を横断して集約します。",
    }
    sections = (
        ("Run overview", render_kv_table(metadata)),
        ("What finished?", render_table(("group", "stage", "required", "present", "total", "status", "description_ja"), group_summary)),
        ("Prediction runs", render_table(("run_id", "predictions_exists", "prediction_rows", "metrics_exists", "predictions_path", "metrics_path"), run_rows)),
        ("Main metric winners", render_table(("prediction_level", "target_name", "best_run_id", "score_name", "score", "n_available"), winner_rows)),
        ("Metrics by run and target", render_table(("run_id", "prediction_level", "target_name", "n_available", "n_skipped", "metrics", "skip_reasons"), metric_rows)),
        ("Visual evidence artifacts", render_table(("artifact", "exists", "rows_or_files", "path"), visual_rows)),
        ("Clip lifecycle", render_table(("clip_status", "count"), clip_rows)),
        ("Artifact inventory", render_table(("group", "artifact", "exists", "kind", "rows_or_files", "size_mb", "status", "path"), inventory)),
    )
    html = render_page(
        "Full Pipeline Dashboard",
        report_id,
        sections,
        subtitle="One entry point for Colab results: artifact completeness, model metrics, clip/CV outputs, and where the main files live.",
    )
    write_page(output_path, html)
    summary_payload = {
        "schema_version": "pipeline_dashboard_v1",
        "metadata": metadata,
        "artifact_group_summary": group_summary,
        "prediction_runs": run_rows,
        "metric_winners": winner_rows,
        "visual_evidence": visual_rows,
        "clip_lifecycle": clip_rows,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"pipeline_dashboard": output_path, "pipeline_dashboard_summary": summary_path}
