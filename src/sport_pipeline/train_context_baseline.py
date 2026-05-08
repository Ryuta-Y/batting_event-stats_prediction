"""Run the lightweight context-only baseline and evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sport_pipeline.artifact_check import write_json
from sport_pipeline.build_context_dataset import build_context_dataset_artifact, build_context_dataset_rows
from sport_pipeline.evaluation import evaluate_predictions, load_target_registry, validate_prediction_rows
from sport_pipeline.io import read_table, write_table
from sport_pipeline.models.context import CatBoostContextBaseline, ConstantContextBaseline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET_REGISTRY = PROJECT_ROOT / "configs/targets/target_registry_v1.yaml"


def run_context_baseline(
    base_dir: str | Path,
    *,
    bbe_events: str | Path | None = None,
    target_registry: str | Path = DEFAULT_TARGET_REGISTRY,
    run_id: str = "context_constant_mean_smoke",
    model_family: str = "constant_mean",
    catboost_task_type: str = "CPU",
    catboost_devices: str | None = None,
    output_suffix: str = ".parquet",
    resume: bool = True,
) -> dict[str, Path]:
    """Build context dataset, predictions_v1, and metrics_v1."""

    base = Path(base_dir)
    bbe_path = Path(bbe_events) if bbe_events is not None else base / "manifests/bbe_events_v1.parquet"
    context_path = base / f"datasets/context_dataset_v1/manifest{output_suffix}"
    prediction_path = base / f"predictions/{run_id}/predictions_v1{output_suffix}"
    metrics_path = base / f"predictions/{run_id}/metrics_v1.json"
    progress_path = base / f"reports/preflight/train_context_baseline_{run_id}_progress.json"

    def write_progress(status: str, **extra: object) -> None:
        payload = {
            "schema_version": "train_context_baseline_progress_v1",
            "status": status,
            "run_id": run_id,
            "model_family": model_family,
            "resume": resume,
            "bbe_events": str(bbe_path),
            "outputs": {
                "context_dataset": str(context_path),
                "predictions": str(prediction_path),
                "metrics": str(metrics_path),
                "progress": str(progress_path),
            },
        }
        payload.update(extra)
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if resume and context_path.exists() and prediction_path.exists() and metrics_path.exists():
        write_progress("reused_existing")
        return {
            "context_dataset": context_path,
            "predictions": prediction_path,
            "metrics": metrics_path,
            "progress": progress_path,
        }

    write_progress("loading_bbe")
    bbe_rows = read_table(bbe_path)
    context_rows = build_context_dataset_rows(bbe_rows)
    write_progress("writing_context_dataset", bbe_rows=len(bbe_rows), context_rows=len(context_rows))
    context_path = build_context_dataset_artifact(base, bbe_path, output_suffix=output_suffix)
    targets = load_target_registry(target_registry)

    def catboost_progress(status: str, **extra: object) -> None:
        write_progress(status, bbe_rows=len(bbe_rows), context_rows=len(context_rows), **extra)

    if model_family == "constant_mean":
        baseline = ConstantContextBaseline(targets)
    elif model_family == "catboost":
        baseline = CatBoostContextBaseline(
            targets,
            task_type=catboost_task_type,
            devices=catboost_devices,
            progress_callback=catboost_progress,
        )
    else:
        raise ValueError(f"unsupported context model_family: {model_family}")
    write_progress("training", bbe_rows=len(bbe_rows), context_rows=len(context_rows))
    baseline.fit(context_rows)
    write_progress("predicting", bbe_rows=len(bbe_rows), context_rows=len(context_rows))
    predictions = baseline.predict_rows(context_rows, run_id=run_id)
    validate_prediction_rows(predictions)
    write_progress("evaluating", bbe_rows=len(bbe_rows), context_rows=len(context_rows), prediction_rows=len(predictions))
    metrics = evaluate_predictions(predictions, targets, run_id=run_id)
    write_table(prediction_path, predictions)
    write_json(metrics, metrics_path)
    write_progress(
        "complete",
        bbe_rows=len(bbe_rows),
        context_rows=len(context_rows),
        prediction_rows=len(predictions),
    )
    return {
        "context_dataset": context_path,
        "predictions": prediction_path,
        "metrics": metrics_path,
        "progress": progress_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run context-only constant baseline.")
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/baseball_vision")
    parser.add_argument("--bbe-events", default=None)
    parser.add_argument("--target-registry", default=str(DEFAULT_TARGET_REGISTRY))
    parser.add_argument("--run-id", default="context_constant_mean_smoke")
    parser.add_argument(
        "--model-family",
        choices=("constant_mean", "catboost"),
        default="constant_mean",
        help="constant_mean is local-safe; catboost is the stronger Colab-only context baseline.",
    )
    parser.add_argument("--output-format", choices=("parquet", "jsonl", "json", "csv"), default="parquet")
    parser.add_argument("--catboost-task-type", choices=("CPU", "GPU"), default="CPU")
    parser.add_argument("--catboost-devices", default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    outputs = run_context_baseline(
        args.base_dir,
        bbe_events=args.bbe_events,
        target_registry=args.target_registry,
        run_id=args.run_id,
        model_family=args.model_family,
        catboost_task_type=args.catboost_task_type,
        catboost_devices=args.catboost_devices,
        output_suffix="." + args.output_format,
        resume=not args.no_resume,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
