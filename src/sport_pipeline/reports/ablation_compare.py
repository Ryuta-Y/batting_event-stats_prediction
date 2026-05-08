"""Ablation comparison reports for raw video, frozen video, and pose/CV models."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

from sport_pipeline.artifact_check import write_json
from sport_pipeline.evaluation import evaluate_predictions, load_target_registry
from sport_pipeline.io import read_table
from sport_pipeline.reports.build_static import read_metrics_payload
from sport_pipeline.reports.html import render_kv_table, render_page, render_table, write_page


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TARGET_REGISTRY = PROJECT_ROOT / "configs/targets/target_registry_v1.yaml"

DEFAULT_RUN_SPECS = (
    {
        "run_id": "context_catboost_mlb_2024_2026_v2",
        "label": "context CatBoost",
        "modality": "tabular_context",
        "representation": "Statcast/context only",
        "trainable_part": "CatBoost",
        "question_axis": "context_only_baseline",
    },
    {
        "run_id": "video_lightweight_cv2_mlb_2024_2026_v2",
        "label": "raw video lightweight",
        "modality": "raw_video",
        "representation": "RGB clip statistics",
        "trainable_part": "deterministic lightweight head",
        "question_axis": "is_any_video_signal_present",
    },
    {
        "run_id": "video_frozen_encoder_mlb_2024_2026_v2",
        "label": "raw video VideoMAE frozen",
        "modality": "raw_video",
        "representation": "VideoMAE frozen contact clip embedding",
        "trainable_part": "stat heads only",
        "question_axis": "does_pretrained_video_embedding_help",
    },
    {
        "run_id": "video_raw_finetune_mlb_2024_2026_v2",
        "label": "raw video fine-tune",
        "modality": "raw_video",
        "representation": "contact-aligned RGB frames",
        "trainable_part": "R3D-18 or tiny 3D CNN + stat heads",
        "question_axis": "does_end_to_end_video_learning_help",
    },
    {
        "run_id": "sequence_tcn_mlb_2024_2026_v2",
        "label": "pose/object structured TCN",
        "modality": "pose_cv_sequence",
        "representation": "pose/object/bat/plate structured sequence",
        "trainable_part": "TCN",
        "question_axis": "does_pose_compression_keep_enough_information",
    },
    {
        "run_id": "fusion_mlb_2024_2026_v2",
        "label": "late fusion",
        "modality": "fusion",
        "representation": "available context/sequence/video predictions",
        "trainable_part": "late fusion",
        "question_axis": "does_combining_signals_help",
    },
)

PRIMARY_METRICS = ("mae", "brier", "rmse", "f1", "r2", "spearman")
HIGHER_IS_BETTER = {"f1", "r2", "spearman"}


def run_specs_from_profile(run_profile: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Build ablation run specs from the active run profile instead of static ids."""

    run_ids = run_profile.get("run_ids", {})
    replacements = {
        "context_only_baseline": run_ids.get("context_run_id"),
        "is_any_video_signal_present": run_ids.get("video_lightweight_run_id"),
        "does_pretrained_video_embedding_help": run_ids.get("video_frozen_run_id") or run_ids.get("video_run_id"),
        "does_end_to_end_video_learning_help": run_ids.get("video_finetune_run_id"),
        "does_pose_compression_keep_enough_information": run_ids.get("sequence_tcn_run_id"),
        "does_combining_signals_help": run_ids.get("fusion_run_id"),
    }
    specs = []
    for spec in DEFAULT_RUN_SPECS:
        resolved = replacements.get(str(spec.get("question_axis")))
        if not resolved:
            continue
        item = dict(spec)
        item["run_id"] = str(resolved)
        specs.append(item)
    return tuple(specs)


def _metric_path(base_dir: Path, run_id: str) -> Path:
    return base_dir / "predictions" / run_id / "metrics_v1.json"


def _prediction_path(base_dir: Path, run_id: str) -> Path:
    return base_dir / "predictions" / run_id / "predictions_v1.parquet"


def _prediction_path_candidates(base_dir: Path, run_id: str) -> tuple[Path, ...]:
    prediction_dir = base_dir / "predictions" / run_id
    return (
        prediction_dir / "predictions_v1.parquet",
        prediction_dir / "predictions_v1.jsonl",
        prediction_dir / "predictions_v1.json",
        prediction_dir / "predictions_v1.csv",
    )


def _existing_prediction_path(base_dir: Path, run_id: str) -> Path | None:
    for path in _prediction_path_candidates(base_dir, run_id):
        if path.exists():
            return path
    return None


def _primary_metric(metrics: dict[str, Any]) -> tuple[str | None, float | None, bool]:
    for name in PRIMARY_METRICS:
        value = metrics.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return name, float(value), name in HIGHER_IS_BETTER
    return None, None, False


def _rows_from_metrics_payload(
    payload: dict[str, Any],
    spec: dict[str, Any],
    path: Path,
    *,
    split: str = "all",
) -> list[dict[str, Any]]:
    run_id = str(spec["run_id"])
    rows: list[dict[str, Any]] = []
    by_level = payload.get("metrics") or {}
    for prediction_level, by_target in by_level.items():
        if not isinstance(by_target, dict):
            continue
        for target_name, metric_values in by_target.items():
            if not isinstance(metric_values, dict):
                continue
            metric_name, metric_value, higher_is_better = _primary_metric(metric_values)
            rows.append(
                {
                    "run_id": run_id,
                    "label": spec.get("label", run_id),
                    "modality": spec.get("modality", "unknown"),
                    "representation": spec.get("representation", "unknown"),
                    "trainable_part": spec.get("trainable_part", "unknown"),
                    "question_axis": spec.get("question_axis", "unknown"),
                    "split": split,
                    "prediction_level": prediction_level,
                    "target_name": target_name,
                    "primary_metric": metric_name,
                    "primary_value": metric_value,
                    "higher_is_better": higher_is_better,
                    "n_available": metric_values.get("n_available"),
                    "n_skipped": metric_values.get("n_skipped"),
                    "mae": metric_values.get("mae"),
                    "rmse": metric_values.get("rmse"),
                    "brier": metric_values.get("brier"),
                    "f1": metric_values.get("f1"),
                    "r2": metric_values.get("r2"),
                    "spearman": metric_values.get("spearman"),
                    "metrics_path": str(path),
                }
            )
    return rows


def _flatten_metrics(base_dir: Path, run_specs: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for spec in run_specs:
        run_id = str(spec["run_id"])
        path = _metric_path(base_dir, run_id)
        if not path.exists():
            missing.append(
                {
                    "run_id": run_id,
                    "label": spec.get("label", run_id),
                    "modality": spec.get("modality", "unknown"),
                    "missing_path": str(path),
                    "status": "missing_metrics",
                }
            )
            continue
        payload = read_metrics_payload(path)
        rows.extend(_rows_from_metrics_payload(payload, spec, path))
    return rows, missing


def _flatten_split_metrics(
    base_dir: Path,
    run_specs: Iterable[dict[str, Any]],
    target_registry: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    targets = load_target_registry(target_registry)
    for spec in run_specs:
        run_id = str(spec["run_id"])
        path = _existing_prediction_path(base_dir, run_id)
        if path is None:
            missing.append(
                {
                    "run_id": run_id,
                    "label": spec.get("label", run_id),
                    "modality": spec.get("modality", "unknown"),
                    "missing_path": str(_prediction_path(base_dir, run_id)),
                    "status": "missing_predictions_for_split_metrics",
                }
            )
            continue
        try:
            prediction_rows = read_table(path)
            by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in prediction_rows:
                by_split[str(row.get("split") or "unknown")].append(row)
            for split, split_rows in sorted(by_split.items()):
                payload = evaluate_predictions(split_rows, targets, run_id=f"{run_id}__{split}")
                rows.extend(_rows_from_metrics_payload(payload, spec, path, split=split))
        except Exception as exc:  # pragma: no cover - defensive for user-created artifacts
            missing.append(
                {
                    "run_id": run_id,
                    "label": spec.get("label", run_id),
                    "modality": spec.get("modality", "unknown"),
                    "missing_path": str(path),
                    "status": f"split_metrics_error: {exc}",
                }
            )
    return rows, missing


def _best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("primary_value") is None:
            continue
        grouped.setdefault(
            (str(row.get("split", "all")), str(row["prediction_level"]), str(row["target_name"])),
            [],
        ).append(row)
    best = []
    for (split, level, target), candidates in sorted(grouped.items()):
        higher = bool(candidates[0].get("higher_is_better"))
        chosen = sorted(candidates, key=lambda row: float(row["primary_value"]), reverse=higher)[0]
        best.append(
            {
                "split": split,
                "prediction_level": level,
                "target_name": target,
                "best_run": chosen["label"],
                "best_run_id": chosen["run_id"],
                "modality": chosen["modality"],
                "primary_metric": chosen["primary_metric"],
                "primary_value": chosen["primary_value"],
                "higher_is_better": higher,
                "n_available": chosen["n_available"],
            }
        )
    return best


def _pairwise_question_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_target: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("primary_value") is None:
            continue
        by_target[
            (
                str(row["question_axis"]),
                str(row.get("split", "all")),
                str(row["prediction_level"]),
                str(row["target_name"]),
            )
        ] = row

    pairs = (
        (
            "Does fine-tuning raw video help over frozen VideoMAE?",
            "does_pretrained_video_embedding_help",
            "does_end_to_end_video_learning_help",
        ),
        (
            "Does pose/CV compression keep enough information vs frozen raw video?",
            "does_pretrained_video_embedding_help",
            "does_pose_compression_keep_enough_information",
        ),
        (
            "Does pose/CV compression keep enough information vs fine-tuned raw video?",
            "does_end_to_end_video_learning_help",
            "does_pose_compression_keep_enough_information",
        ),
        (
            "Is any video signal better than context only?",
            "context_only_baseline",
            "does_pretrained_video_embedding_help",
        ),
    )
    output = []
    targets = sorted({(split, level, target) for (_axis, split, level, target) in by_target})
    for question, baseline_run, candidate_run in pairs:
        for split, level, target in targets:
            baseline = by_target.get((baseline_run, split, level, target))
            candidate = by_target.get((candidate_run, split, level, target))
            if baseline is None or candidate is None:
                continue
            if baseline.get("primary_metric") != candidate.get("primary_metric"):
                continue
            baseline_value = float(baseline["primary_value"])
            candidate_value = float(candidate["primary_value"])
            higher = bool(candidate.get("higher_is_better"))
            improvement = candidate_value - baseline_value if higher else baseline_value - candidate_value
            better = "candidate" if improvement > 0 else ("baseline" if improvement < 0 else "tie")
            output.append(
                {
                    "question": question,
                    "split": split,
                    "prediction_level": level,
                    "target_name": target,
                    "metric": candidate["primary_metric"],
                    "baseline": baseline["label"],
                    "baseline_value": baseline_value,
                    "candidate": candidate["label"],
                    "candidate_value": candidate_value,
                    "improvement_positive_is_candidate_better": improvement,
                    "better": better,
                }
            )
    return output


def build_video_ablation_report(
    base_dir: str | Path,
    *,
    report_id: str = "video_ablation_mlb_2024_2026_v2",
    run_specs: Iterable[dict[str, Any]] = DEFAULT_RUN_SPECS,
    target_registry: str | Path = DEFAULT_TARGET_REGISTRY,
) -> dict[str, Path]:
    """Write a static ablation report for the current Drive artifacts."""

    base = Path(base_dir)
    specs = [dict(spec) for spec in run_specs]
    rows, missing = _flatten_metrics(base, specs)
    split_rows, split_missing = _flatten_split_metrics(base, specs, target_registry)
    best = _best_rows(rows)
    split_best = _best_rows(split_rows)
    pairwise = _pairwise_question_rows(rows)
    split_pairwise = _pairwise_question_rows(split_rows)
    output_dir = base / "reports" / "ablation_compare" / report_id
    outputs = {
        "html": output_dir / "index.html",
        "summary": output_dir / "summary.json",
    }
    metadata = {
        "schema_version": "video_ablation_report_v1",
        "base_dir": str(base),
        "report_id": report_id,
        "metric_rows": len(rows),
        "split_metric_rows": len(split_rows),
        "missing_runs": len(missing),
        "missing_split_inputs": len(split_missing),
        "target_registry": str(target_registry),
        "note": "Overall metrics mirror each run's metrics_v1.json. Split metrics are recomputed from predictions_v1 grouped by split when predictions include split.",
    }
    html = render_page(
        "Video Ablation Compare",
        report_id,
        (
            ("Inputs", render_kv_table(metadata)),
            (
                "Question Map",
                render_table(
                    ("label", "run_id", "modality", "representation", "trainable_part", "question_axis"),
                    specs,
                ),
            ),
            (
                "Metrics",
                render_table(
                    (
                        "split",
                        "label",
                        "target_name",
                        "prediction_level",
                        "primary_metric",
                        "primary_value",
                        "n_available",
                        "mae",
                        "rmse",
                        "brier",
                        "f1",
                        "r2",
                        "spearman",
                    ),
                    rows,
                ),
            ),
            (
                "Best By Target",
                render_table(
                    ("split", "target_name", "prediction_level", "best_run", "modality", "primary_metric", "primary_value", "n_available"),
                    best,
                ),
            ),
            (
                "Direct Questions",
                render_table(
                    (
                        "question",
                        "split",
                        "target_name",
                        "metric",
                        "baseline",
                        "baseline_value",
                        "candidate",
                        "candidate_value",
                        "improvement_positive_is_candidate_better",
                        "better",
                    ),
                    pairwise,
                ),
            ),
            (
                "Metrics By Split",
                render_table(
                    (
                        "split",
                        "label",
                        "target_name",
                        "prediction_level",
                        "primary_metric",
                        "primary_value",
                        "n_available",
                        "mae",
                        "rmse",
                        "brier",
                        "f1",
                        "r2",
                        "spearman",
                    ),
                    split_rows,
                ),
            ),
            (
                "Best By Target And Split",
                render_table(
                    ("split", "target_name", "prediction_level", "best_run", "modality", "primary_metric", "primary_value", "n_available"),
                    split_best,
                ),
            ),
            (
                "Direct Questions By Split",
                render_table(
                    (
                        "question",
                        "split",
                        "target_name",
                        "metric",
                        "baseline",
                        "baseline_value",
                        "candidate",
                        "candidate_value",
                        "improvement_positive_is_candidate_better",
                        "better",
                    ),
                    split_pairwise,
                ),
            ),
            ("Missing Runs", render_table(("label", "run_id", "modality", "missing_path", "status"), missing)),
            (
                "Missing Split Inputs",
                render_table(("label", "run_id", "modality", "missing_path", "status"), split_missing),
            ),
        ),
        subtitle="Raw RGB video, frozen video embeddings, tiny raw-video fine-tuning, and pose/object structured sequence are compared target by target.",
    )
    write_page(outputs["html"], html)
    write_json(
        {
            **metadata,
            "run_specs": specs,
            "metrics_by_run": rows,
            "metrics_by_split": split_rows,
            "best_by_target": best,
            "best_by_target_split": split_best,
            "direct_questions": pairwise,
            "direct_questions_by_split": split_pairwise,
            "missing_run_rows": missing,
            "missing_split_input_rows": split_missing,
            "outputs": {key: str(path) for key, path in outputs.items()},
        },
        outputs["summary"],
    )
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build video ablation comparison report.")
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/baseball_vision")
    parser.add_argument("--report-id", default="video_ablation_mlb_2024_2026_v2")
    parser.add_argument("--target-registry", default=str(DEFAULT_TARGET_REGISTRY))
    args = parser.parse_args(argv)
    outputs = build_video_ablation_report(args.base_dir, report_id=args.report_id, target_registry=args.target_registry)
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
