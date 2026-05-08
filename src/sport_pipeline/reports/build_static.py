"""Build Phase 9 static HTML reports from Colab/Drive artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from sport_pipeline.io import read_table
from sport_pipeline.reports.html import render_cards, render_kv_table, render_page, render_table, write_page
from sport_pipeline.reports.summaries import (
    clip_quality_counts,
    ensemble_prior_summary,
    experiment_compare_summary,
    failure_cases,
    ops_unavailable_reasons,
    target_availability_summary,
)


REPORT_CONTRACT_VERSION = "static_reports_v1"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_table_artifact(path: str | Path) -> list[dict[str, Any]]:
    """Read JSON/JSONL/Parquet table artifacts through shared normalization.

    Keeping report IO on the same code path as manifest/model IO prevents
    pandas/Arrow nullable values from leaking into schema validation as NaN.
    """

    return read_table(path)


def read_metrics_payload(path: str | Path) -> dict[str, Any]:
    """Read metrics_v1 JSON."""

    payload = _read_json(Path(path))
    if not isinstance(payload, dict):
        raise ValueError("metrics artifact must be a JSON object")
    return payload


def report_paths(output_root: str | Path, run_id: str) -> dict[str, Path]:
    """Return standard report output paths."""

    root = Path(output_root)
    return {
        "target_availability": root / "target_availability" / run_id / "index.html",
        "experiment_compare": root / "experiment_compare" / run_id / "index.html",
        "failure_browser": root / "failure_browser" / run_id / "index.html",
        "clip_quality": root / "clip_quality" / run_id / "index.html",
    }


def _group_rows_by_level(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        level = str(row.get("prediction_level") or "unknown")
        grouped.setdefault(level, []).append(row)
    return {level: grouped[level] for level in sorted(grouped)}


def build_static_reports(
    *,
    base_dir: str | Path,
    run_id: str,
    prediction_rows: Iterable[dict[str, Any]],
    metrics_payloads: Iterable[dict[str, Any]] | None = None,
    clip_rows: Iterable[dict[str, Any]] | None = None,
    fusion_audit_rows: Iterable[dict[str, Any]] | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Path]:
    """Build Phase 9 static HTML reports and return generated paths."""

    base_path = Path(base_dir)
    report_root = Path(output_root) if output_root is not None else base_path / "reports"
    predictions = list(prediction_rows)
    metrics = list(metrics_payloads or [])
    clips = list(clip_rows or [])
    audit = list(fusion_audit_rows or [])
    paths = report_paths(report_root, run_id)

    metadata = {
        "schema_version": REPORT_CONTRACT_VERSION,
        "base_dir": str(base_path),
        "run_id": run_id,
        "config_hash": "unknown",
        "code_version": "unknown",
        "prediction_rows": len(predictions),
        "metrics_payloads": len(metrics),
        "clip_rows": len(clips),
        "fusion_audit_rows": len(audit),
    }

    availability = target_availability_summary(predictions)
    ops_reasons = ops_unavailable_reasons(predictions)
    availability_html = render_page(
        "Target Availability Report",
        run_id,
        (
            ("Inputs", render_kv_table(metadata)),
            ("Target availability", render_table(("target_name", "prediction_level", "available", "missing", "missing_reasons"), availability)),
            ("OPS unavailable reasons", render_table(("reason", "count"), ops_reasons)),
        ),
        subtitle="Missing xBA/xwOBA labels stay visible and OPS is skipped unless player-season PA data is available.",
    )
    write_page(paths["target_availability"], availability_html)

    compare_rows = experiment_compare_summary(metrics)
    compare_sections: list[tuple[str, str]] = [("Inputs", render_kv_table(metadata))]
    for level, rows_for_level in _group_rows_by_level(compare_rows).items():
        compare_sections.append(
            (
                f"Metrics by run and target: {level}",
                render_table(("run_id", "prediction_level", "target_name", "n_available", "n_skipped", "metrics", "skip_reasons"), rows_for_level),
            )
        )
    if not compare_rows:
        compare_sections.append(("Metrics by run and target", render_table(("run_id", "prediction_level", "target_name", "n_available", "n_skipped", "metrics", "skip_reasons"), [])))
    compare_sections.append(
        (
            "Same-event ensemble vs player-season prior",
            render_table(("comparison_axis", "rows", "available_rows", "scopes"), ensemble_prior_summary(predictions, audit)),
        )
    )
    compare_html = render_page(
        "Experiment Comparison Report",
        run_id,
        tuple(compare_sections),
        subtitle="Model comparison is target-by-target; event predictions are not averaged across different events.",
    )
    write_page(paths["experiment_compare"], compare_html)

    clip_summary = clip_quality_counts(clips)
    clip_html = render_page(
        "Clip Quality Report",
        run_id,
        (
            ("Inputs", render_kv_table(metadata)),
            ("Clip lifecycle counts", render_table(("clip_status", "count", "quality_tiers", "reasons"), clip_summary)),
        ),
        subtitle="Clean clip, review_only, and excluded rows remain separate cohorts.",
    )
    write_page(paths["clip_quality"], clip_html)

    failures = failure_cases(predictions, clips, limit=50)
    failure_sections: list[tuple[str, str]] = [
        ("Inputs", render_kv_table(metadata)),
        ("Filters and sort", render_kv_table({"sort": "absolute error descending, then missing labels", "limit": 50, "large_video_policy": "links only; videos are not copied into the repo"}))
    ]
    for level, rows_for_level in _group_rows_by_level(failures).items():
        failure_sections.append((f"Failure cases: {level}", render_cards(rows_for_level)))
    if not failures:
        failure_sections.append(("Failure cases", render_cards([])))
    failure_html = render_page(
        "Failure Browser",
        run_id,
        tuple(failure_sections),
        subtitle="Static browser for high-error rows, missing labels, clip metadata, overlays, and prior/ensemble metadata.",
    )
    write_page(paths["failure_browser"], failure_html)
    return paths


def _dummy_predictions(run_id: str) -> list[dict[str, Any]]:
    base = {
        "run_id": run_id,
        "batter_season_id": "111_2026",
        "target_source": "statcast",
        "head_kind": "regression",
        "loss_name": "huber",
        "prior_mode": "none",
        "requires_pa_manifest": False,
        "n_prior_clips": 0,
        "aggregation_method": "single_clip",
        "same_event_ensemble": False,
        "prediction_std": None,
    }
    return [
        {
            **base,
            "sample_id": "game123_ab45_p3__ev",
            "event_id": "game123_ab45_p3",
            "prediction_level": "event",
            "target_name": "ev",
            "y_true": 101.2,
            "y_pred": 96.5,
            "target_available": True,
            "aggregation_scope": "current_event_only",
            "label_missing_reason": None,
        },
        {
            **base,
            "sample_id": "game123_ab45_p3__xba",
            "event_id": "game123_ab45_p3",
            "prediction_level": "event",
            "target_name": "xba",
            "y_true": None,
            "y_pred": None,
            "target_available": False,
            "head_kind": "probability",
            "loss_name": "mse",
            "aggregation_scope": "current_event_only",
            "label_missing_reason": "statcast_expected_outcome_missing",
        },
        {
            **base,
            "sample_id": "111_2026__ops",
            "event_id": None,
            "prediction_level": "player_season",
            "target_name": "ops",
            "y_true": None,
            "y_pred": None,
            "target_available": False,
            "aggregation_scope": "player_season_aggregate",
            "requires_pa_manifest": True,
            "label_missing_reason": "pa_manifest_unavailable",
        },
        {
            **base,
            "sample_id": "game123_ab45_p3__ev__ensemble",
            "event_id": "game123_ab45_p3",
            "prediction_level": "event",
            "target_name": "ev",
            "y_true": 101.2,
            "y_pred": 99.8,
            "target_available": True,
            "aggregation_scope": "same_event_view_crop_augmentation_ensemble",
            "label_missing_reason": None,
            "same_event_ensemble": True,
            "prediction_std": 1.25,
        },
        {
            **base,
            "sample_id": "game123_ab45_p3__ev__prior",
            "event_id": "game123_ab45_p3",
            "prediction_level": "event",
            "target_name": "ev",
            "y_true": 101.2,
            "y_pred": 100.3,
            "target_available": True,
            "aggregation_scope": "current_event_with_player_season_prior",
            "label_missing_reason": None,
            "prior_mode": "past_only",
            "n_prior_clips": 3,
            "aggregation_method": "quality_weighted_pooling",
        },
    ]


def _dummy_metrics(run_id: str) -> dict[str, Any]:
    return {
        "schema_version": "metrics_v1",
        "run_id": run_id,
        "metrics": {
            "event": {"ev": {"mae": 2.0, "rmse": 2.6, "n_available": 3, "n_skipped": 0}},
            "player_season": {},
        },
        "label_availability": {"xba": {"available": 0, "missing": 1}, "ops": {"available": 0, "missing": 1}},
        "skipped": {"xba": {"statcast_expected_outcome_missing": 1}, "ops": {"pa_manifest_unavailable": 1}},
    }


def _dummy_clips() -> list[dict[str, Any]]:
    return [
        {
            "clip_id": "clip_clean",
            "event_id": "game123_ab45_p3",
            "clip_status": "clean_clip",
            "quality_tier": "usable_primary",
            "view_label": "bat_catcher_view",
            "clip_path": "clips/smoke/clip_clean.mp4",
            "overlay_path": "debug/smoke/clip_clean_overlay.mp4",
        },
        {
            "clip_id": "clip_review",
            "event_id": "game124_ab12_p1",
            "clip_status": "review_only",
            "quality_tier": "review_only",
            "review_reason": "replay_or_cutaway",
        },
        {
            "clip_id": "clip_excluded",
            "event_id": "game999_ab1_p1",
            "clip_status": "excluded",
            "quality_tier": "excluded",
            "exclusion_reason": "wrong_event",
        },
    ]


def _dummy_audit(run_id: str) -> list[dict[str, Any]]:
    return [
        {
            "fusion_run_id": run_id,
            "source_aggregation_scope": "same_event_view_crop_augmentation_ensemble",
            "source_prior_mode": "none",
            "source_target_available": True,
        },
        {
            "fusion_run_id": run_id,
            "source_aggregation_scope": "current_event_with_player_season_prior",
            "source_prior_mode": "past_only",
            "source_target_available": True,
        },
    ]


def run_report_smoke(base_dir: str | Path, run_id: str = "phase9_smoke") -> dict[str, Path]:
    """Generate tiny static reports for local or Colab preflight validation."""

    return build_static_reports(
        base_dir=base_dir,
        run_id=run_id,
        prediction_rows=_dummy_predictions(run_id),
        metrics_payloads=[_dummy_metrics(run_id)],
        clip_rows=_dummy_clips(),
        fusion_audit_rows=_dummy_audit(run_id),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build static report HTML pages.")
    parser.add_argument("--base-dir", required=True, help="Drive artifact root, usually /content/drive/MyDrive/baseball_vision")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--predictions", help="predictions_v1 table path, JSON/JSONL/Parquet")
    parser.add_argument("--metrics", action="append", default=[], help="metrics_v1 JSON path; may be repeated")
    parser.add_argument("--clips", help="clips_v1 table path, JSON/JSONL/Parquet")
    parser.add_argument("--fusion-audit", help="fusion_input_audit_v1 table path, JSON/JSONL/Parquet")
    parser.add_argument("--output-root", help="override report output root")
    parser.add_argument("--smoke", action="store_true", help="write tiny dummy reports instead of reading artifacts")
    args = parser.parse_args(argv)

    if args.smoke:
        paths = run_report_smoke(args.base_dir, args.run_id)
    else:
        if not args.predictions:
            parser.error("--predictions is required unless --smoke is set")
        paths = build_static_reports(
            base_dir=args.base_dir,
            run_id=args.run_id,
            prediction_rows=read_table_artifact(args.predictions),
            metrics_payloads=[read_metrics_payload(path) for path in args.metrics],
            clip_rows=read_table_artifact(args.clips) if args.clips else [],
            fusion_audit_rows=read_table_artifact(args.fusion_audit) if args.fusion_audit else [],
            output_root=args.output_root,
        )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
