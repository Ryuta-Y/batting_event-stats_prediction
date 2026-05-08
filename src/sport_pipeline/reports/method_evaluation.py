"""Method-level evaluation reports that exclude context-dominated fusion rows.

This report is for the research question: how well does each visual/mechanics
representation predict event outcomes and player-season aggregates before late
fusion? It intentionally reads upstream ``predictions_v1`` artifacts directly.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from sport_pipeline.artifact_check import write_json
from sport_pipeline.evaluation import evaluate_predictions, load_target_registry, validate_prediction_rows
from sport_pipeline.io import read_table
from sport_pipeline.pipeline.run_profile import run_id
from sport_pipeline.reports.html import html_escape, render_kv_table, render_page, render_table, write_page


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TARGET_REGISTRY = PROJECT_ROOT / "configs/targets/target_registry_v1.yaml"
EVENT_TARGETS = ("ev", "la", "hard_hit", "barrel", "xba", "xwoba")
PLAYER_SEASON_TARGETS = ("avg_ev", "avg_la", "avg_xba", "avg_xwoba", "hard_hit_rate", "barrel_rate", "ba", "ops", "obp", "slg")
PRIMARY_METRICS = ("mae", "brier", "rmse", "f1", "r2", "spearman")
HIGHER_IS_BETTER = {"f1", "r2", "spearman"}


VISUAL_SCOPE_ALLOWLIST = {
    "current_event_only",
    "current_event_with_player_season_prior",
    "current_event_structured_sequence",
    "raw_video_lightweight",
    "video_frozen_encoder",
    "image_frozen_encoder",
    "raw_video_finetune",
    "player_season_mechanics_prior",
    "vlm_mechanics_features",
}


DEFAULT_METHOD_EXPLANATIONS = {
    "context": {
        "label": "Context baseline",
        "method_family": "tabular_context",
        "input_signal": "Statcast/game/count/pitch/context columns, no batting video pixels",
        "what_it_tests_ja": "映像を見ない時にどこまで当たるか。動画系の下限/比較対象。",
        "aggregation_scope": "context_only",
    },
    "structured_sequence": {
        "label": "Structured sequence deterministic",
        "method_family": "pose_object_sequence",
        "input_signal": "clip quality, contact timing, deterministic T x D sequence features",
        "what_it_tests_ja": "検出/姿勢の前に、clip metadata だけで作る mechanics prior がどこまで効くか。",
        "aggregation_scope": "current_event_with_player_season_prior",
    },
    "pose_object_tcn": {
        "label": "Detection/Tracking/Pose TCN",
        "method_family": "pose_object_sequence",
        "input_signal": "YOLO detections, ByteTrack ids, pose skeletons, bat line, plate homography over time",
        "what_it_tests_ja": "検出・追跡・棒人間風 pose/バット/ホームベースを圧縮した時系列だけで予測できるか。",
        "aggregation_scope": "current_event_structured_sequence",
    },
    "raw_video_lightweight": {
        "label": "Raw video lightweight",
        "method_family": "raw_video",
        "input_signal": "OpenCV RGB/motion statistics from contact-aligned clips",
        "what_it_tests_ja": "DNN なしの単純な動画統計に信号があるか。",
        "aggregation_scope": "raw_video_lightweight",
    },
    "raw_video_frozen": {
        "label": "Raw video frozen encoder",
        "method_family": "raw_video",
        "input_signal": "VideoMAE/DINO frozen embeddings plus lightweight supervised heads",
        "what_it_tests_ja": "事前学習済み動画/画像 encoder に batting clip を入れるだけで信号が出るか。",
        "aggregation_scope": "video_frozen_encoder",
    },
    "raw_video_finetune": {
        "label": "Raw video DNN fine-tune",
        "method_family": "raw_video",
        "input_signal": "contact-aligned RGB frames into tiny 3D CNN or R3D-18 style model",
        "what_it_tests_ja": "単純に動画を DNN に入れて end-to-end 学習すると改善するか。",
        "aggregation_scope": "raw_video_finetune",
    },
    "player_season": {
        "label": "Player-season mechanics prior",
        "method_family": "player_season",
        "input_signal": "multi-clip batter-season mechanics embedding",
        "what_it_tests_ja": "単打席ではなく、同一選手シーズンの複数 clip から選手 stats を予測できるか。",
        "aggregation_scope": "player_season_mechanics_prior",
    },
    "vlm": {
        "label": "VLM mechanics features",
        "method_family": "vlm",
        "input_signal": "VLM caption / mechanics tags / visual scores extracted from batting clips or contact frames",
        "what_it_tests_ja": "VLM が言語化した stance/load/swing/follow-through などが予測に使えるか。",
        "aggregation_scope": "vlm_mechanics_features",
    },
}


def _prediction_path(base_dir: Path, run_id_value: str) -> Path | None:
    run_dir = base_dir / "predictions" / run_id_value
    for suffix in (".parquet", ".jsonl", ".json", ".csv"):
        path = run_dir / f"predictions_v1{suffix}"
        if path.exists():
            return path
    return None


def _projection_run_id(run_id_value: str) -> str:
    return f"{run_id_value}_player_season_projection"


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    row_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in row_list:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_list)


def _primary_metric(values: dict[str, Any]) -> tuple[str | None, float | None, bool]:
    for metric_name in PRIMARY_METRICS:
        value = values.get(metric_name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return metric_name, float(value), metric_name in HIGHER_IS_BETTER
    return None, None, False


def _flatten_metrics(payload: dict[str, Any], spec: dict[str, Any], *, metric_scope: str) -> list[dict[str, Any]]:
    rows = []
    metrics = payload.get("metrics") or {}
    availability = payload.get("label_availability") or {}
    skipped = payload.get("skipped") or {}
    for level, by_target in metrics.items():
        if not isinstance(by_target, dict):
            continue
        for target_name, values in by_target.items():
            if not isinstance(values, dict):
                continue
            metric_name, metric_value, higher = _primary_metric(values)
            availability_row = availability.get(target_name, {}) if isinstance(availability, dict) else {}
            rows.append(
                {
                    "metric_scope": metric_scope,
                    "method_key": spec["method_key"],
                    "label": spec["label"],
                    "run_id": spec["run_id"],
                    "method_family": spec["method_family"],
                    "prediction_level": level,
                    "target_name": target_name,
                    "primary_metric": metric_name,
                    "primary_value": metric_value,
                    "higher_is_better": higher,
                    "n_available": values.get("n_available", 0),
                    "n_skipped": values.get("n_skipped", 0),
                    "n_missing": availability_row.get("missing", 0),
                    "skip_reasons": json.dumps(skipped.get(target_name, {}), ensure_ascii=False) if isinstance(skipped, dict) else "{}",
                    "mae": values.get("mae"),
                    "rmse": values.get("rmse"),
                    "r2": values.get("r2"),
                    "spearman": values.get("spearman"),
                    "f1": values.get("f1"),
                    "brier": values.get("brier"),
                }
            )
    return rows


def _prediction_key(row: dict[str, Any]) -> tuple[str, str, str] | None:
    level = str(row.get("prediction_level"))
    target_name = str(row.get("target_name"))
    if level == "event":
        event_id = row.get("event_id")
        if event_id is None:
            return None
        return (level, str(event_id), target_name)
    if level == "player_season":
        batter_season_id = row.get("batter_season_id")
        if batter_season_id is None:
            return None
        return (level, str(batter_season_id), target_name)
    return None


def _filter_visual_rows(rows: list[dict[str, Any]], *, include_context: bool) -> list[dict[str, Any]]:
    if include_context:
        return rows
    output = []
    for row in rows:
        if str(row.get("aggregation_scope")) == "context_only":
            continue
        output.append(row)
    return output


def _sample_count_rows(spec: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        key = (str(row.get("prediction_level")), str(row.get("target_name")))
        item = grouped.setdefault(key, {"prediction_rows": 0, "available_rows": 0, "unique_samples": 0})
        item["prediction_rows"] += 1
        if row.get("target_available") and row.get("y_pred") is not None:
            item["available_rows"] += 1
    unique_by_group: dict[tuple[str, str], set[tuple[str, str, str]]] = {}
    for row in rows:
        key = (str(row.get("prediction_level")), str(row.get("target_name")))
        prediction_key = _prediction_key(row)
        if prediction_key is not None:
            unique_by_group.setdefault(key, set()).add(prediction_key)
    output = []
    for (level, target_name), counts in sorted(grouped.items()):
        output.append(
            {
                "method_key": spec["method_key"],
                "label": spec["label"],
                "run_id": spec["run_id"],
                "method_family": spec["method_family"],
                "prediction_level": level,
                "target_name": target_name,
                "prediction_rows": counts["prediction_rows"],
                "available_rows": counts["available_rows"],
                "unique_samples": len(unique_by_group.get((level, target_name), set())),
            }
        )
    return output


def _intersection_metrics(
    method_rows: dict[str, list[dict[str, Any]]],
    method_specs: list[dict[str, Any]],
    targets: dict[str, Any],
    *,
    min_methods: int = 2,
) -> list[dict[str, Any]]:
    keys_by_method_target: dict[tuple[str, str, str], set[tuple[str, str, str]]] = {}
    row_by_method_key: dict[tuple[str, tuple[str, str, str]], dict[str, Any]] = {}
    for method_key, rows in method_rows.items():
        for row in rows:
            if not row.get("target_available") or row.get("y_pred") is None:
                continue
            key = _prediction_key(row)
            if key is None:
                continue
            level, _sample, target_name = key
            keys_by_method_target.setdefault((method_key, level, target_name), set()).add(key)
            row_by_method_key[(method_key, key)] = row

    spec_by_key = {spec["method_key"]: spec for spec in method_specs}
    output: list[dict[str, Any]] = []
    target_groups = sorted({(level, target_name) for (_method, level, target_name) in keys_by_method_target})
    for level, target_name in target_groups:
        candidate_methods = [
            method_key
            for method_key in spec_by_key
            if (method_key, level, target_name) in keys_by_method_target
        ]
        if len(candidate_methods) < min_methods:
            continue
        common_keys = set.intersection(*(keys_by_method_target[(method_key, level, target_name)] for method_key in candidate_methods))
        if not common_keys:
            continue
        for method_key in candidate_methods:
            spec = spec_by_key[method_key]
            subset = [row_by_method_key[(method_key, key)] for key in sorted(common_keys)]
            payload = evaluate_predictions(subset, targets, run_id=f"{spec['run_id']}__intersection")
            rows = _flatten_metrics(payload, spec, metric_scope="same_sample_intersection")
            for row in rows:
                if row["prediction_level"] == level and row["target_name"] == target_name:
                    row["intersection_methods"] = ",".join(candidate_methods)
                    row["intersection_samples"] = len(common_keys)
                    output.append(row)
    return output


def _best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("primary_value") is None:
            continue
        grouped.setdefault((str(row["metric_scope"]), str(row["prediction_level"]), str(row["target_name"])), []).append(row)
    output = []
    for (scope, level, target_name), candidates in sorted(grouped.items()):
        higher = bool(candidates[0].get("higher_is_better"))
        chosen = sorted(candidates, key=lambda item: float(item["primary_value"]), reverse=higher)[0]
        output.append(
            {
                "metric_scope": scope,
                "prediction_level": level,
                "target_name": target_name,
                "best_label": chosen["label"],
                "best_run_id": chosen["run_id"],
                "method_family": chosen["method_family"],
                "primary_metric": chosen["primary_metric"],
                "primary_value": chosen["primary_value"],
                "n_available": chosen["n_available"],
            }
        )
    return output


def _method_specs_from_profile(
    run_profile: dict[str, Any],
    *,
    include_context: bool,
    include_fusion: bool,
) -> list[dict[str, Any]]:
    candidates = [
        ("context", "context_run_id", include_context),
        ("structured_sequence", "sequence_run_id", True),
        ("pose_object_tcn", "sequence_tcn_run_id", True),
        ("raw_video_lightweight", "video_lightweight_run_id", True),
        ("raw_video_frozen", "video_frozen_run_id", True),
        ("raw_video_finetune", "video_finetune_run_id", True),
        ("player_season", "player_season_run_id", True),
        ("vlm", "vlm_run_id", True),
        ("fusion", "fusion_run_id", include_fusion),
    ]
    specs = []
    for method_key, run_key, enabled in candidates:
        if not enabled:
            continue
        try:
            resolved_run_id = run_id(run_profile, run_key)
        except KeyError:
            continue
        explanation = DEFAULT_METHOD_EXPLANATIONS.get(
            method_key,
            {
                "label": method_key,
                "method_family": "unknown",
                "input_signal": "unknown",
                "what_it_tests_ja": "",
                "aggregation_scope": "",
            },
        )
        spec = {"method_key": method_key, "run_id": resolved_run_id, **explanation}
        if method_key == "fusion":
            spec.update(
                {
                    "label": "Late fusion",
                    "method_family": "fusion",
                    "input_signal": "weighted average of available upstream prediction rows",
                    "what_it_tests_ja": "前段手法を混ぜた時に最終性能が上がるか。ただし context_only が大半になり得るので主表からは通常外す。",
                    "aggregation_scope": "late_fusion_event/player_season",
                }
            )
        specs.append(spec)
    return specs


def _svg_bar_chart(path: Path, rows: list[dict[str, Any]], *, title: str, value_key: str, label_key: str = "label") -> str | None:
    values = [(str(row.get(label_key)), row.get(value_key)) for row in rows if isinstance(row.get(value_key), (int, float))]
    if not values:
        return None
    values = values[:12]
    width = 980
    bar_h = 26
    gap = 10
    left = 250
    top = 56
    height = top + len(values) * (bar_h + gap) + 30
    max_value = max(float(value) for _label, value in values) or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="32" font-family="Arial" font-size="22" font-weight="700" fill="#17202a">{html_escape(title)}</text>',
    ]
    for index, (label, value) in enumerate(values):
        y = top + index * (bar_h + gap)
        bar_w = int((float(value) / max_value) * (width - left - 90))
        parts.append(f'<text x="24" y="{y + 18}" font-family="Arial" font-size="13" fill="#17202a">{html_escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{bar_w}" height="{bar_h}" rx="3" fill="#0f766e"/>')
        parts.append(f'<text x="{left + bar_w + 8}" y="{y + 18}" font-family="Arial" font-size="13" fill="#17202a">{float(value):.4g}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return str(path)


def build_method_evaluation_report(
    base_dir: str | Path,
    run_profile: dict[str, Any],
    *,
    report_id: str | None = None,
    target_registry: str | Path = DEFAULT_TARGET_REGISTRY,
    include_context_reference: bool = False,
    include_fusion_reference: bool = False,
) -> dict[str, Path]:
    """Build method-level outputs from upstream visual/mechanics predictions."""

    base = Path(base_dir)
    resolved_report_id = report_id or run_id(run_profile, "method_evaluation_report_id", "method_evaluation_mlb_2024_2026_v2")
    output_root = base / "reports" / "method_evaluation" / resolved_report_id
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    outputs = {
        "html": output_root / "index.html",
        "summary": output_root / "summary.json",
        "method_metrics": tables_dir / "method_metrics.csv",
        "same_sample_metrics": tables_dir / "same_sample_intersection_metrics.csv",
        "sample_counts": tables_dir / "sample_counts.csv",
        "method_map": tables_dir / "method_map.csv",
        "best_by_target": tables_dir / "best_by_target.csv",
        "availability_chart": figures_dir / "available_rows_by_method.svg",
    }
    targets = load_target_registry(target_registry)
    method_specs = _method_specs_from_profile(
        run_profile,
        include_context=include_context_reference,
        include_fusion=include_fusion_reference,
    )
    method_rows: dict[str, list[dict[str, Any]]] = {}
    missing: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    sample_counts: list[dict[str, Any]] = []
    for spec in method_specs:
        path = _prediction_path(base, spec["run_id"])
        if path is None:
            missing.append({**spec, "status": "missing_predictions", "path": str(base / "predictions" / spec["run_id"] / "predictions_v1.parquet")})
            method_rows[spec["method_key"]] = []
            continue
        rows = read_table(path)
        validate_prediction_rows(rows)
        rows = _filter_visual_rows(rows, include_context=include_context_reference)
        projection_id = _projection_run_id(spec["run_id"])
        projection_path = _prediction_path(base, projection_id)
        if projection_path is not None:
            projection_rows = read_table(projection_path)
            validate_prediction_rows(projection_rows)
            rows.extend(projection_rows)
            spec["player_season_projection_run_id"] = projection_id
            spec["player_season_projection_path"] = str(projection_path)
        method_rows[spec["method_key"]] = rows
        sample_counts.extend(_sample_count_rows(spec, rows))
        if not rows:
            missing.append({**spec, "status": "no_rows_after_filter", "path": str(path)})
            continue
        payload = evaluate_predictions(rows, targets, run_id=spec["run_id"])
        metrics_rows.extend(_flatten_metrics(payload, spec, metric_scope="per_method_all_available"))

    same_sample_metrics = _intersection_metrics(method_rows, method_specs, targets)
    all_metric_rows = metrics_rows + same_sample_metrics
    best_rows = _best_rows(all_metric_rows)
    chart_rows = sorted(
        [
            {
                "label": row["label"],
                "available_rows": row["available_rows"],
            }
            for row in sample_counts
            if row["prediction_level"] == "event" and row["target_name"] == "ev"
        ],
        key=lambda item: int(item["available_rows"]),
        reverse=True,
    )
    _svg_bar_chart(outputs["availability_chart"], chart_rows, title="Available Event Samples (EV)", value_key="available_rows")

    method_map_rows = [
        {
            "method_key": spec["method_key"],
            "label": spec["label"],
            "run_id": spec["run_id"],
            "method_family": spec["method_family"],
            "input_signal": spec["input_signal"],
            "aggregation_scope": spec["aggregation_scope"],
            "player_season_projection_run_id": spec.get("player_season_projection_run_id"),
            "what_it_tests_ja": spec["what_it_tests_ja"],
        }
        for spec in method_specs
    ]
    _write_csv(outputs["method_metrics"], metrics_rows)
    _write_csv(outputs["same_sample_metrics"], same_sample_metrics)
    _write_csv(outputs["sample_counts"], sample_counts)
    _write_csv(outputs["method_map"], method_map_rows)
    _write_csv(outputs["best_by_target"], best_rows)

    metadata = {
        "schema_version": "method_evaluation_report_v1",
        "base_dir": str(base),
        "report_id": resolved_report_id,
        "target_registry": str(target_registry),
        "include_context_reference": include_context_reference,
        "include_fusion_reference": include_fusion_reference,
        "method_count": len(method_specs),
        "metric_rows": len(metrics_rows),
        "same_sample_metric_rows": len(same_sample_metrics),
        "missing_inputs": len(missing),
        "note_ja": "この report は fusion ではなく前段の predictions_v1 を直接評価する。標準では context_only を主表から除外する。",
    }
    html = render_page(
        "Method Evaluation",
        resolved_report_id,
        (
            ("Inputs", render_kv_table(metadata)),
            (
                "Method Map",
                render_table(
                    ("label", "run_id", "method_family", "input_signal", "aggregation_scope", "player_season_projection_run_id", "what_it_tests_ja"),
                    method_map_rows,
                ),
            ),
            (
                "Sample Counts",
                render_table(
                    ("label", "prediction_level", "target_name", "prediction_rows", "available_rows", "unique_samples"),
                    sample_counts,
                ),
            ),
            (
                "Per-Method Metrics",
                render_table(
                    (
                        "label",
                        "prediction_level",
                        "target_name",
                        "primary_metric",
                        "primary_value",
                        "n_available",
                        "mae",
                        "rmse",
                        "r2",
                        "spearman",
                        "f1",
                        "brier",
                    ),
                    metrics_rows,
                ),
            ),
            (
                "Same-Sample Intersection Metrics",
                render_table(
                    (
                        "label",
                        "prediction_level",
                        "target_name",
                        "primary_metric",
                        "primary_value",
                        "intersection_samples",
                        "mae",
                        "rmse",
                        "r2",
                        "spearman",
                        "f1",
                        "brier",
                    ),
                    same_sample_metrics,
                ),
            ),
            (
                "Best By Target",
                render_table(
                    ("metric_scope", "prediction_level", "target_name", "best_label", "method_family", "primary_metric", "primary_value", "n_available"),
                    best_rows,
                ),
            ),
            ("Missing Or Empty Inputs", render_table(("label", "run_id", "method_family", "status", "path"), missing)),
        ),
        subtitle="Method outputs before/after late fusion. Player-season rows include direct player-season models and optional projections from event-level predictions.",
    )
    write_page(outputs["html"], html)
    write_json(
        {
            **metadata,
            "method_map": method_map_rows,
            "sample_counts": sample_counts,
            "method_metrics": metrics_rows,
            "same_sample_intersection_metrics": same_sample_metrics,
            "best_by_target": best_rows,
            "missing": missing,
            "outputs": {key: str(path) for key, path in outputs.items()},
        },
        outputs["summary"],
    )
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build visual/mechanics method-level evaluation report.")
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/baseball_vision")
    parser.add_argument("--run-profile", default=str(PROJECT_ROOT / "configs/runs/mlb_2024_2026_real_colab_v2.json"))
    parser.add_argument("--report-id", default=None)
    parser.add_argument("--target-registry", default=str(DEFAULT_TARGET_REGISTRY))
    parser.add_argument("--include-context-reference", action="store_true")
    parser.add_argument("--include-fusion-reference", action="store_true")
    args = parser.parse_args(argv)
    profile = json.loads(Path(args.run_profile).read_text(encoding="utf-8"))
    outputs = build_method_evaluation_report(
        args.base_dir,
        profile,
        report_id=args.report_id,
        target_registry=args.target_registry,
        include_context_reference=args.include_context_reference,
        include_fusion_reference=args.include_fusion_reference,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
