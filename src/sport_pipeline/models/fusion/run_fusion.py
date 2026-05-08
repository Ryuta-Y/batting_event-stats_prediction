"""Run late fusion over upstream predictions_v1 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from sport_pipeline.artifact_check import write_json
from sport_pipeline.evaluation import evaluate_predictions, load_target_registry, validate_prediction_rows
from sport_pipeline.io import read_table, write_table
from sport_pipeline.models.fusion.contracts import FUSION_INPUT_AUDIT_SCHEMA
from sport_pipeline.models.fusion.late_fusion import late_fuse_prediction_rows, learn_validation_scope_weights
from sport_pipeline.schemas.data_manifest import validate_rows


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TARGET_REGISTRY = PROJECT_ROOT / "configs/targets/target_registry_v1.yaml"
DEFAULT_SOURCE_RUNS = (
    "context_catboost_mlb_2024_2026_v1",
    "sequence_tcn_mlb_2024_2026_v1",
    "video_frozen_encoder_mlb_2024_2026_v1",
)
DEFAULT_FUSION_CONFIG = PROJECT_ROOT / "configs/models/fusion/late_fusion_v1.json"


def _prediction_suffix(path: Path) -> str:
    return path.suffix if path.suffix else ".parquet"


def _default_prediction_path(base_dir: Path, run_id: str) -> Path:
    for suffix in (".parquet", ".jsonl", ".json", ".csv"):
        path = base_dir / f"predictions/{run_id}/predictions_v1{suffix}"
        if path.exists():
            return path
    return base_dir / f"predictions/{run_id}/predictions_v1.parquet"


def _read_prediction_sources(base_dir: Path, source_runs: Iterable[str]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    status: list[dict] = []
    for run_id in source_runs:
        path = _default_prediction_path(base_dir, run_id)
        if not path.exists():
            status.append({"run_id": run_id, "path": str(path), "exists": False, "rows": 0})
            continue
        source_rows = read_table(path)
        validate_prediction_rows(source_rows)
        rows.extend(source_rows)
        status.append({"run_id": run_id, "path": str(path), "exists": True, "rows": len(source_rows)})
    return rows, status


def _load_scope_weights(config_path: str | Path | None) -> dict[str, float] | None:
    if config_path is None:
        return None
    path = Path(config_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    weights = payload.get("scope_weights")
    if weights is None:
        return None
    if not isinstance(weights, dict):
        raise ValueError(f"scope_weights must be a mapping in {path}")
    return {str(key): float(value) for key, value in weights.items()}


def run_full_fusion(
    base_dir: str | Path,
    *,
    fusion_run_id: str = "fusion_mlb_2024_2026_v1",
    source_runs: Iterable[str] = DEFAULT_SOURCE_RUNS,
    target_registry: str | Path = DEFAULT_TARGET_REGISTRY,
    fusion_config: str | Path | None = DEFAULT_FUSION_CONFIG,
    learn_weights_from_validation: bool = False,
    output_suffix: str = ".parquet",
) -> dict[str, Path]:
    """Fuse available upstream predictions without crossing event boundaries."""

    base = Path(base_dir)
    rows, source_status = _read_prediction_sources(base, source_runs)
    scope_weights = _load_scope_weights(fusion_config)
    learned_weights = learn_validation_scope_weights(rows) if learn_weights_from_validation else {}
    if learned_weights:
        scope_weights = {**(scope_weights or {}), **learned_weights}
    result = late_fuse_prediction_rows(rows, fusion_run_id=fusion_run_id, scope_weights=scope_weights) if rows else None
    prediction_rows = result.prediction_rows if result is not None else []
    audit_rows = result.audit_rows if result is not None else []
    validate_prediction_rows(prediction_rows)
    validate_rows(FUSION_INPUT_AUDIT_SCHEMA, audit_rows)
    targets = load_target_registry(target_registry)
    metrics = evaluate_predictions(prediction_rows, targets, run_id=fusion_run_id)

    outputs = {
        "predictions": base / f"predictions/{fusion_run_id}/predictions_v1{output_suffix}",
        "metrics": base / f"predictions/{fusion_run_id}/metrics_v1.json",
        "fusion_input_audit": base / f"predictions/{fusion_run_id}/fusion_input_audit_v1{output_suffix}",
        "summary": base / f"reports/preflight/full_fusion_{fusion_run_id}.json",
    }
    write_table(outputs["predictions"], prediction_rows)
    write_json(metrics, outputs["metrics"])
    write_table(outputs["fusion_input_audit"], audit_rows)
    write_json(
        {
            "schema_version": "full_fusion_summary_v1",
            "fusion_run_id": fusion_run_id,
            "source_status": source_status,
            "fusion_config": None if fusion_config is None else str(fusion_config),
            "scope_weights": scope_weights,
            "learn_weights_from_validation": learn_weights_from_validation,
            "learned_weights": learned_weights,
            "input_prediction_rows": len(rows),
            "fused_prediction_rows": len(prediction_rows),
            "audit_rows": len(audit_rows),
            "output_suffix": output_suffix,
        },
        outputs["summary"],
    )
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run full late fusion over upstream predictions_v1 artifacts.")
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/baseball_vision")
    parser.add_argument("--fusion-run-id", default="fusion_mlb_2024_2026_v1")
    parser.add_argument(
        "--source-run",
        action="append",
        dest="source_runs",
        default=None,
        help="Upstream prediction run id. Repeat for multiple runs.",
    )
    parser.add_argument("--target-registry", default=str(DEFAULT_TARGET_REGISTRY))
    parser.add_argument("--fusion-config", default=str(DEFAULT_FUSION_CONFIG))
    parser.add_argument("--learn-weights-from-validation", action="store_true")
    parser.add_argument("--output-format", choices=("parquet", "jsonl", "json", "csv"), default="parquet")
    args = parser.parse_args(argv)
    outputs = run_full_fusion(
        args.base_dir,
        fusion_run_id=args.fusion_run_id,
        source_runs=args.source_runs or DEFAULT_SOURCE_RUNS,
        target_registry=args.target_registry,
        fusion_config=args.fusion_config,
        learn_weights_from_validation=args.learn_weights_from_validation,
        output_suffix="." + args.output_format,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
