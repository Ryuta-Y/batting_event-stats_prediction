"""Paper-style research outputs for the full batting vision Colab run.

The functions here read existing Drive artifacts and write a separate report
bundle under ``reports/research_outputs``. They do not mutate source artifacts.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from sport_pipeline.pipeline.run_profile import artifact_namespace, run_id as profile_run_id
from sport_pipeline.reports.html import html_escape, render_kv_table, render_page, render_table, write_page
from sport_pipeline.reports.run_selection import report_run_candidates


EVENT_TARGET_ORDER = ("ev", "la", "hard_hit", "barrel", "xba", "xwoba")
REGRESSION_TARGETS = {"ev", "la", "xba", "xwoba"}
BINARY_TARGETS = {"hard_hit", "barrel"}
EXPECTED_COLAB_BASE_DIR = "/content/drive/MyDrive/baseball_vision"


def _status(ok: bool) -> str:
    return "ok" if ok else "warning"


@dataclass(frozen=True)
class ResearchOutputPaths:
    root: Path
    figures: Path
    tables: Path
    videos: Path
    index_html: Path
    summary_json: Path


def _ensure_dirs(paths: ResearchOutputPaths) -> None:
    for path in (paths.root, paths.figures, paths.tables, paths.videos):
        path.mkdir(parents=True, exist_ok=True)


def _paths(output_root: Path, report_id: str) -> ResearchOutputPaths:
    root = output_root / "research_outputs" / report_id
    return ResearchOutputPaths(
        root=root,
        figures=root / "figures",
        tables=root / "tables",
        videos=root / "videos",
        index_html=root / "index.html",
        summary_json=root / "summary.json",
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _version_ok(value: Any, version: str = "_v2") -> bool:
    return version in str(value or "")


def _fusion_source_runs(run_profile: dict[str, Any]) -> list[str]:
    fusion_settings = (run_profile.get("execution") or {}).get("fusion") or {}
    source_runs = fusion_settings.get("source_runs") or []
    return [str(run_id) for run_id in source_runs if run_id]


def _model_design_rows(run_profile: dict[str, Any]) -> list[dict[str, Any]]:
    run_ids = run_profile.get("run_ids", {})
    execution = run_profile.get("execution", {})
    vlm = execution.get("vlm_mechanics", {})
    frozen = execution.get("frozen_visual_encoder", {})
    raw = execution.get("raw_video_finetune", {})
    sequence = execution.get("sequence_tcn", {})
    fusion = execution.get("fusion", {})
    return [
        {
            "method": "Context CatBoost",
            "run_id": run_ids.get("context_run_id"),
            "level": "event",
            "inputs": "Statcast BBE context features only",
            "model": "CatBoost/tabular baseline",
            "targets": "EV, LA, hard-hit, barrel, optional xBA/xwOBA",
            "notes": "No video evidence; leakage-aware reference baseline.",
        },
        {
            "method": "Structured sequence deterministic",
            "run_id": run_ids.get("sequence_run_id"),
            "level": "event",
            "inputs": "Contact-aligned clip features, player-season prior when available",
            "model": "Deterministic structured feature baseline",
            "targets": "Event Statcast heads",
            "notes": "Uses same-event clip features without averaging different BBE events.",
        },
        {
            "method": "Sequence TCN",
            "run_id": run_ids.get("sequence_tcn_run_id"),
            "level": "event",
            "inputs": "YOLO/tracking/pose/bat-line structured frame sequence",
            "model": f"TCN depth={sequence.get('depth')} hidden={sequence.get('hidden_dim')}",
            "targets": "Event Statcast heads",
            "notes": f"prior_feature_mode={sequence.get('prior_feature_mode')}",
        },
        {
            "method": "Lightweight video CV",
            "run_id": run_ids.get("video_lightweight_run_id"),
            "level": "event",
            "inputs": "OpenCV lightweight motion/appearance features",
            "model": "Classical CV feature head",
            "targets": "Event Statcast heads",
            "notes": "Fast video baseline for comparison.",
        },
        {
            "method": "Frozen visual encoder",
            "run_id": run_ids.get("video_frozen_run_id"),
            "level": "event",
            "inputs": "Contact-aligned clip frames",
            "model": f"{frozen.get('encoder', 'videomae')} / {frozen.get('model_id', '')}",
            "targets": "Event Statcast heads",
            "notes": "Frozen embedding plus lightweight supervised head.",
        },
        {
            "method": "Raw video fine-tune",
            "run_id": run_ids.get("video_finetune_run_id"),
            "level": "event",
            "inputs": "Raw contact-aligned clip frames",
            "model": f"{raw.get('model_family')} pretrained={raw.get('pretrained')} epochs={raw.get('max_epochs')}",
            "targets": "Event Statcast heads",
            "notes": "End-to-end video baseline; heavy stage runs on GPU.",
        },
        {
            "method": "Player-season mechanics prior",
            "run_id": run_ids.get("player_season_run_id"),
            "level": "player_season",
            "inputs": "Multiple clips aggregated per batter-season plus batting labels",
            "model": "Player-season aggregate baseline",
            "targets": "OPS, OBP, SLG, BA, average EV/LA/xBA/xwOBA, hard-hit/barrel rates",
            "notes": "OPS/OBP/SLG are season-level targets, not event-level BBE heads.",
        },
        {
            "method": "VLM mechanics",
            "run_id": run_ids.get("vlm_run_id"),
            "level": "event and projected player_season",
            "inputs": f"Qwen VLM captions/tags from {vlm.get('caption_max_rows')} clips",
            "model": vlm.get("hf_model_id", "open-source VLM"),
            "targets": "Event Statcast heads via VLM feature baseline",
            "notes": f"input_mode={vlm.get('input_mode')} reader={vlm.get('video_reader_backend')} fallback_debug_frame={vlm.get('fallback_to_debug_frame')}",
        },
        {
            "method": "Late fusion",
            "run_id": run_ids.get("fusion_run_id"),
            "level": "event and player_season",
            "inputs": ", ".join(_fusion_source_runs(run_profile)),
            "model": "Weighted average over aligned predictions by event/player-season and target",
            "targets": "All available event and player-season targets",
            "notes": f"learn_weights_from_validation={fusion.get('learn_weights_from_validation', False)}",
        },
    ]


def _fusion_source_status_rows(base_dir: Path, final_run_id: str) -> list[dict[str, Any]]:
    summary = _read_json(base_dir / f"reports/preflight/full_fusion_{final_run_id}.json")
    rows = summary.get("source_status") or []
    return [row for row in rows if isinstance(row, dict)]


def _method_evaluation_root(base_dir: Path, run_profile: dict[str, Any]) -> Path:
    run_ids = run_profile.get("run_ids", {})
    report_id = str(run_ids.get("method_evaluation_report_id") or "method_evaluation_mlb_2024_2026_v2")
    return base_dir / "reports" / "method_evaluation" / report_id


def _provenance_checks(
    base_dir: Path,
    run_profile: dict[str, Any],
    full_run_id: str,
    final_run_id: str,
    fusion_source_status: list[dict[str, Any]],
) -> dict[str, Any]:
    paths = run_profile.get("paths", {})
    run_ids = {key: str(value) for key, value in (run_profile.get("run_ids") or {}).items() if value}
    namespace = {key: str(value) for key, value in artifact_namespace(run_profile).items() if value}
    source_runs = _fusion_source_runs(run_profile)
    source_status_by_run = {str(row.get("run_id")): row for row in fusion_source_status}
    fusion_summary_runs = [run for run in source_status_by_run if run and run != "None"]
    critical_ids = list(run_ids.values()) + list(namespace.values()) + source_runs + [full_run_id, final_run_id]
    checks = [
        {
            "check": "base_dir_is_baseball_vision",
            "status": _status(str(base_dir) == EXPECTED_COLAB_BASE_DIR),
            "value": str(base_dir),
            "expected": EXPECTED_COLAB_BASE_DIR,
        },
        {
            "check": "run_profile_base_dir_matches",
            "status": _status(str(paths.get("base_dir")) in {"", str(base_dir)} or paths.get("base_dir") is None),
            "value": str(paths.get("base_dir")),
            "expected": str(base_dir),
        },
        {
            "check": "full_run_id_is_v2",
            "status": _status(_version_ok(full_run_id)),
            "value": full_run_id,
            "expected": "contains _v2",
        },
        {
            "check": "final_fusion_run_id_is_v2",
            "status": _status(_version_ok(final_run_id)),
            "value": final_run_id,
            "expected": "contains _v2",
        },
        {
            "check": "all_configured_run_ids_are_v2",
            "status": _status(all(_version_ok(value) for value in run_ids.values())),
            "value": ", ".join(f"{key}={value}" for key, value in sorted(run_ids.items())),
            "expected": "every configured run_id contains _v2",
        },
        {
            "check": "report_ids_are_v2",
            "status": _status(
                _version_ok(run_ids.get("method_evaluation_report_id"))
                and _version_ok(run_ids.get("video_ablation_report_id"))
            ),
            "value": f"method_evaluation_report_id={run_ids.get('method_evaluation_report_id')}, video_ablation_report_id={run_ids.get('video_ablation_report_id')}",
            "expected": "report ids contain _v2",
        },
        {
            "check": "all_artifact_namespaces_are_v2",
            "status": _status(all(_version_ok(value) for value in namespace.values())),
            "value": ", ".join(f"{key}={value}" for key, value in sorted(namespace.items())),
            "expected": "every artifact namespace contains _v2",
        },
        {
            "check": "no_v1_substrings_in_critical_ids",
            "status": _status(all("_v1" not in value for value in critical_ids)),
            "value": ", ".join(value for value in critical_ids if "_v1" in value),
            "expected": "no configured run/artifact/report id contains _v1",
        },
        {
            "check": "fusion_source_runs_are_v2",
            "status": _status(bool(source_runs) and all(_version_ok(value) and "_v1" not in value for value in source_runs)),
            "value": ", ".join(source_runs),
            "expected": "all fusion source_runs contain _v2 and no _v1",
        },
        {
            "check": "fusion_summary_source_runs_match_config",
            "status": _status(bool(source_runs) and set(source_runs).issubset(set(fusion_summary_runs))),
            "value": ", ".join(fusion_summary_runs),
            "expected": "fusion summary source_status includes every configured fusion source_run",
        },
        {
            "check": "fusion_summary_source_runs_are_v2",
            "status": _status(bool(fusion_summary_runs) and all(_version_ok(value) and "_v1" not in value for value in fusion_summary_runs)),
            "value": ", ".join(fusion_summary_runs),
            "expected": "fusion summary source_status run ids contain _v2 and no _v1",
        },
        {
            "check": "fusion_config_includes_vlm",
            "status": _status(any(str(run).startswith(str(run_ids.get("vlm_run_id", "vlm_mechanics"))) for run in source_runs)),
            "value": ", ".join(source_runs),
            "expected": str(run_ids.get("vlm_run_id")),
        },
        {
            "check": "fusion_summary_includes_vlm_inputs",
            "status": _status(any(str(run).startswith(str(run_ids.get("vlm_run_id", "vlm_mechanics"))) and bool(source_status_by_run.get(str(run), {}).get("exists")) for run in source_runs)),
            "value": ", ".join(
                f"{row.get('run_id')} exists={row.get('exists')} rows={row.get('rows')}"
                for row in fusion_source_status
                if str(row.get("run_id", "")).startswith(str(run_ids.get("vlm_run_id", "vlm_mechanics")))
            ),
            "expected": "VLM prediction and projection sources exist in fusion summary",
        },
    ]
    return {
        "overall_status": "ok" if all(row["status"] == "ok" for row in checks) else "warning",
        "checks": checks,
    }


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(output) or math.isinf(output):
        return None
    return output


def _maybe_num_rows(path: Path) -> int | None:
    if not path.exists() or path.is_dir():
        return None
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    if suffix == ".json":
        payload = _read_json(path)
        if isinstance(payload.get("rows"), list):
            return len(payload["rows"])
        return None
    if suffix == ".csv":
        with path.open("r", encoding="utf-8") as handle:
            return max(0, sum(1 for _ in handle) - 1)
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


def _read_table(path: Path, columns: list[str] | None = None, *, max_rows: int | None = None) -> Any:
    import pandas as pd  # type: ignore

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path, columns=columns)
    elif suffix == ".csv":
        df = pd.read_csv(path, usecols=columns)
    elif suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
        if columns:
            df = df[[column for column in columns if column in df.columns]]
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
        df = pd.DataFrame(rows)
        if columns:
            df = df[[column for column in columns if column in df.columns]]
    else:
        raise ValueError(f"Unsupported table extension: {path}")
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=17)
    return df


def _prediction_path(base_dir: Path, run_id: str) -> Path:
    for suffix in (".parquet", ".jsonl", ".json", ".csv"):
        path = base_dir / "predictions" / run_id / f"predictions_v1{suffix}"
        if path.exists():
            return path
    return base_dir / "predictions" / run_id / "predictions_v1.parquet"


def _metrics_path(base_dir: Path, run_id: str) -> Path:
    return base_dir / "predictions" / run_id / "metrics_v1.json"


def _flatten_metrics(metrics_payload: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    run_id = str(metrics_payload.get("run_id", "unknown"))
    metrics = metrics_payload.get("metrics", {})
    skipped = metrics_payload.get("skipped", {})
    if not isinstance(metrics, dict):
        return output
    for prediction_level, target_metrics in metrics.items():
        if not isinstance(target_metrics, dict):
            continue
        for target_name, values in target_metrics.items():
            if not isinstance(values, dict):
                continue
            row = {
                "run_id": run_id,
                "prediction_level": prediction_level,
                "target_name": target_name,
                "n_available": values.get("n_available", 0),
                "n_skipped": values.get("n_skipped", 0),
                "skip_reasons": skipped.get(target_name, {}) if isinstance(skipped, dict) else {},
            }
            for metric_name in ("mae", "rmse", "r2", "spearman", "f1", "brier"):
                row[metric_name] = values.get(metric_name)
            output.append(row)
    return output


def _collect_metrics(base_dir: Path, run_ids: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_id in run_ids:
        payload = _read_json(_metrics_path(base_dir, run_id))
        rows.extend(_flatten_metrics(payload))
    return rows


def _artifact_rows(base_dir: Path, run_profile: dict[str, Any], full_run_id: str, final_run_id: str) -> list[dict[str, Any]]:
    run_ids = run_profile.get("run_ids", {})
    namespace = artifact_namespace(run_profile)
    artifacts = [
        ("Event manifest", "manifests/bbe_events_v1.parquet"),
        ("Video sources", "manifests/video_sources_v1.parquet"),
        ("Player-season batting stats", "manifests/player_season_batting_v1.parquet"),
        ("Downloaded videos", f"raw_videos/{full_run_id}/download_manifest_v1.parquet"),
        ("Candidate segments", f"clips/{full_run_id}/candidate_segments_v1.parquet"),
        ("Clips", f"clips/{full_run_id}/clips_v1.parquet"),
        ("Clip files", f"clips/{full_run_id}/videos"),
        ("Contact frames", f"debug/{full_run_id}/frames"),
        ("Detections", f"detections/{full_run_id}/detections_v1.parquet"),
        ("Tracks", f"tracks/{full_run_id}/tracks_v1.parquet"),
        ("Pose skeletons", f"pose2d/{full_run_id}/pose2d_v1.parquet"),
        ("Bat or plate objects", f"objects/{full_run_id}/bat_detection_v1.parquet"),
        ("Bat lines", f"objects/{full_run_id}/bat_line_v1.parquet"),
        ("Homography", f"homography/{full_run_id}/homography_v1.parquet"),
        ("Structured sequence manifest", f"features/{namespace['structured_sequence_feature_id']}/manifest.parquet"),
        ("Structured sequence frames", f"features/{namespace['structured_sequence_feature_id']}/frames.parquet"),
        ("Player-season prior", f"datasets/{namespace['event_with_prior_dataset_id']}/manifest.parquet"),
        ("Player-season embeddings", f"features/{namespace['player_season_embedding_feature_id']}/manifest.parquet"),
        ("Lightweight video features", f"features/{namespace.get('video_lightweight_feature_id', 'video_lightweight_features_v1')}/manifest.parquet"),
        ("Frozen video embeddings", f"features/{namespace['video_embedding_feature_id']}/manifest.parquet"),
        ("Image embeddings", f"features/{namespace['image_embedding_feature_id']}/manifest.parquet"),
        ("VLM mechanics features", f"features/{namespace.get('vlm_feature_id', 'vlm_mechanics_mlb_2024_2026_v2')}/manifest.parquet"),
        ("HF VLM captioning summary", f"reports/preflight/hf_vlm_captioning_{namespace.get('vlm_feature_id', 'vlm_mechanics_mlb_2024_2026_v2')}.json"),
        ("Final predictions", f"predictions/{final_run_id}/predictions_v1.parquet"),
        ("Final metrics", f"predictions/{final_run_id}/metrics_v1.json"),
        ("Fusion audit", f"predictions/{final_run_id}/fusion_input_audit_v1.parquet"),
        ("CV overlay manifest", f"debug/{full_run_id}/cv_overlay_videos_v1.parquet"),
        ("CV overlay videos", f"debug/{full_run_id}/cv_overlays"),
    ]
    for key in (
        "context_run_id",
        "sequence_run_id",
        "sequence_tcn_run_id",
        "video_lightweight_run_id",
        "video_frozen_run_id",
        "video_finetune_run_id",
        "player_season_run_id",
        "vlm_run_id",
        "fusion_run_id",
    ):
        rid = run_ids.get(key)
        if rid:
            artifacts.append((f"Metrics: {rid}", f"predictions/{rid}/metrics_v1.json"))
            artifacts.append((f"Predictions: {rid}", f"predictions/{rid}/predictions_v1.parquet"))
    method_eval_id = run_ids.get("method_evaluation_report_id")
    if method_eval_id:
        artifacts.extend(
            [
                ("Method evaluation report", f"reports/method_evaluation/{method_eval_id}/index.html"),
                ("Method evaluation summary", f"reports/method_evaluation/{method_eval_id}/summary.json"),
                ("Method evaluation metrics", f"reports/method_evaluation/{method_eval_id}/tables/method_metrics.csv"),
                ("Method evaluation sample counts", f"reports/method_evaluation/{method_eval_id}/tables/sample_counts.csv"),
                ("Method evaluation same-sample metrics", f"reports/method_evaluation/{method_eval_id}/tables/same_sample_intersection_metrics.csv"),
            ]
        )
    rows = []
    for label, relative in artifacts:
        path = base_dir / relative
        if path.is_dir():
            count = sum(1 for _ in path.glob("*"))
        else:
            count = _maybe_num_rows(path)
        rows.append(
            {
                "artifact": label,
                "exists": path.exists(),
                "rows_or_files": count,
                "path": str(path),
            }
        )
    return rows


def _target_availability_rows(metrics_rows: list[dict[str, Any]], final_run_id: str) -> list[dict[str, Any]]:
    rows = [row for row in metrics_rows if row.get("run_id") == final_run_id and row.get("prediction_level") == "event"]
    output = []
    for row in sorted(rows, key=lambda item: EVENT_TARGET_ORDER.index(str(item["target_name"])) if str(item["target_name"]) in EVENT_TARGET_ORDER else 99):
        output.append(
            {
                "target_name": row["target_name"],
                "n_available": row.get("n_available", 0),
                "n_skipped": row.get("n_skipped", 0),
            }
        )
    return output


def _plot_model_metrics(metrics_rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt  # type: ignore
    import pandas as pd  # type: ignore

    if not metrics_rows:
        return
    df = pd.DataFrame(metrics_rows)
    required_columns = {"prediction_level", "target_name", "mae"}
    if not required_columns.issubset(df.columns):
        return
    df = df[(df["prediction_level"] == "event") & (df["target_name"].isin(["ev", "la", "xba", "xwoba"]))]
    if df.empty:
        return
    df["mae"] = pd.to_numeric(df["mae"], errors="coerce")
    df = df.dropna(subset=["mae"])
    if df.empty:
        return
    pivot = df.pivot_table(index="run_id", columns="target_name", values="mae", aggfunc="first")
    pivot = pivot[[column for column in ("ev", "la", "xba", "xwoba") if column in pivot.columns]]
    ax = pivot.plot(kind="bar", figsize=(11, 5.5), width=0.78)
    ax.set_title("Model Comparison by Target")
    ax.set_xlabel("Prediction run")
    ax.set_ylabel("MAE (lower is better)")
    ax.legend(title="Target")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_target_availability(rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    if not rows:
        return
    labels = [str(row["target_name"]).upper() for row in rows]
    available = [int(row.get("n_available") or 0) for row in rows]
    skipped = [int(row.get("n_skipped") or 0) for row in rows]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, available, label="Available labels", color="#0f766e")
    ax.bar(labels, skipped, bottom=available, label="Skipped or missing", color="#94a3b8")
    ax.set_title("Target Label Availability")
    ax.set_xlabel("Target")
    ax.set_ylabel("Rows")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_artifact_counts(rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    visible = [row for row in rows if row.get("rows_or_files") not in (None, "")]
    if not visible:
        return
    labels = [str(row["artifact"]) for row in visible]
    values = [float(row["rows_or_files"]) for row in visible]
    fig, ax = plt.subplots(figsize=(10, max(4.8, len(labels) * 0.32)))
    ax.barh(labels, values, color="#2563eb")
    ax.set_title("Pipeline Artifact Scale")
    ax.set_xlabel("Rows or files")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_fusion_audit(base_dir: Path, final_run_id: str, output_path: Path, table_path: Path) -> list[dict[str, Any]]:
    audit_path = base_dir / "predictions" / final_run_id / "fusion_input_audit_v1.parquet"
    if not audit_path.exists():
        return []
    try:
        df = _read_table(
            audit_path,
            columns=["source_run_id", "source_aggregation_scope", "source_target_available", "fusion_weight"],
        )
    except Exception:
        return []
    rows = []
    for (run, scope), item in df.groupby(["source_run_id", "source_aggregation_scope"], dropna=False):
        rows.append(
            {
                "source_run_id": run,
                "source_aggregation_scope": scope,
                "rows": int(len(item)),
                "available_rows": int(item["source_target_available"].fillna(False).sum()) if "source_target_available" in item else "",
                "mean_fusion_weight": float(item["fusion_weight"].mean()) if "fusion_weight" in item else "",
            }
        )
    _write_csv(table_path, rows)
    try:
        import matplotlib.pyplot as plt  # type: ignore

        counts = Counter(str(value) for value in df["source_aggregation_scope"])
        labels, values = zip(*counts.most_common()) if counts else ([], [])
        if labels:
            fig, ax = plt.subplots(figsize=(9, 4.8))
            ax.barh(labels, values, color="#7c3aed")
            ax.set_title("Fusion Input Provenance")
            ax.set_xlabel("Audit rows")
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=180)
            plt.close(fig)
    except Exception:
        pass
    return rows


def _plot_predictions(base_dir: Path, final_run_id: str, figures_dir: Path, tables_dir: Path, *, max_rows: int) -> dict[str, str]:
    import matplotlib.pyplot as plt  # type: ignore
    import pandas as pd  # type: ignore

    prediction_path = _prediction_path(base_dir, final_run_id)
    if not prediction_path.exists():
        return {}
    columns = ["target_name", "prediction_level", "y_true", "y_pred", "target_available"]
    df = _read_table(prediction_path, columns=columns, max_rows=max_rows)
    if df.empty:
        return {}
    df = df[(df["prediction_level"] == "event") & (df["target_available"].fillna(False))]
    for column in ("y_true", "y_pred"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["y_true", "y_pred"])
    if df.empty:
        return {}
    df["residual"] = df["y_pred"] - df["y_true"]
    df.to_csv(tables_dir / "prediction_plot_sample.csv", index=False)
    scatter_targets = [target for target in ("ev", "la") if target in set(df["target_name"])]
    outputs: dict[str, str] = {}
    if scatter_targets:
        fig, axes = plt.subplots(1, len(scatter_targets), figsize=(5.5 * len(scatter_targets), 5))
        if len(scatter_targets) == 1:
            axes = [axes]
        for ax, target in zip(axes, scatter_targets):
            item = df[df["target_name"] == target]
            if item.empty:
                continue
            ax.scatter(item["y_true"], item["y_pred"], s=8, alpha=0.25, color="#0f766e")
            lo = min(float(item["y_true"].min()), float(item["y_pred"].min()))
            hi = max(float(item["y_true"].max()), float(item["y_pred"].max()))
            ax.plot([lo, hi], [lo, hi], color="#111827", linewidth=1, linestyle="--")
            ax.set_title(f"{target.upper()} Prediction Scatter")
            ax.set_xlabel("Observed")
            ax.set_ylabel("Predicted")
            ax.grid(alpha=0.2)
        fig.tight_layout()
        out = figures_dir / "prediction_scatter_ev_la.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        outputs["prediction_scatter_ev_la"] = str(out)
    if scatter_targets:
        fig, axes = plt.subplots(1, len(scatter_targets), figsize=(5.5 * len(scatter_targets), 4.5))
        if len(scatter_targets) == 1:
            axes = [axes]
        for ax, target in zip(axes, scatter_targets):
            item = df[df["target_name"] == target]
            ax.hist(item["residual"], bins=60, color="#2563eb", alpha=0.82)
            ax.axvline(0, color="#111827", linewidth=1, linestyle="--")
            ax.set_title(f"{target.upper()} Residual Distribution")
            ax.set_xlabel("Prediction error")
            ax.set_ylabel("Rows")
            ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        out = figures_dir / "residual_distribution_ev_la.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        outputs["residual_distribution_ev_la"] = str(out)
    return outputs


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _baseline_comparison_rows(metric_rows: list[dict[str, Any]], baseline_keys: tuple[str, ...] = ("context", "fusion")) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in metric_rows:
        value = _safe_float(row.get("primary_value"))
        if value is None:
            continue
        key = (str(row.get("metric_scope")), str(row.get("prediction_level")), str(row.get("target_name")))
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for (scope, level, target), rows in sorted(grouped.items()):
        for baseline_key in baseline_keys:
            baseline = next((row for row in rows if str(row.get("method_key")) == baseline_key), None)
            if baseline is None:
                continue
            baseline_value = _safe_float(baseline.get("primary_value"))
            if baseline_value is None:
                continue
            for row in rows:
                method_key = str(row.get("method_key"))
                if method_key == baseline_key:
                    continue
                candidate_value = _safe_float(row.get("primary_value"))
                if candidate_value is None:
                    continue
                if str(row.get("primary_metric")) != str(baseline.get("primary_metric")):
                    continue
                higher = _bool_value(row.get("higher_is_better"))
                improvement = candidate_value - baseline_value if higher else baseline_value - candidate_value
                if baseline_value != 0:
                    improvement_pct = improvement / abs(baseline_value)
                else:
                    improvement_pct = None
                output.append(
                    {
                        "metric_scope": scope,
                        "prediction_level": level,
                        "target_name": target,
                        "baseline_method_key": baseline_key,
                        "baseline_label": baseline.get("label"),
                        "baseline_run_id": baseline.get("run_id"),
                        "candidate_method_key": method_key,
                        "candidate_label": row.get("label"),
                        "candidate_run_id": row.get("run_id"),
                        "method_family": row.get("method_family"),
                        "primary_metric": row.get("primary_metric"),
                        "higher_is_better": higher,
                        "baseline_value": baseline_value,
                        "candidate_value": candidate_value,
                        "improvement_positive_is_candidate_better": improvement,
                        "improvement_pct_of_baseline": improvement_pct,
                        "baseline_n_available": baseline.get("n_available"),
                        "candidate_n_available": row.get("n_available"),
                    }
                )
    return output


def _plot_method_sample_counts(sample_counts: list[dict[str, Any]], output_path: Path) -> str | None:
    if not sample_counts:
        return None
    try:
        import pandas as pd  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None
    df = pd.DataFrame(sample_counts)
    if df.empty or "available_rows" not in df:
        return None
    df = df[df["prediction_level"].isin(["event", "player_season"])].copy()
    df["available_rows"] = pd.to_numeric(df["available_rows"], errors="coerce").fillna(0)
    df["target_label"] = df["prediction_level"].astype(str) + ":" + df["target_name"].astype(str)
    pivot = df.pivot_table(index="label", columns="target_label", values="available_rows", aggfunc="sum", fill_value=0)
    if pivot.empty:
        return None
    preferred = [f"event:{target}" for target in EVENT_TARGET_ORDER] + [
        "player_season:avg_ev",
        "player_season:ba",
        "player_season:ops",
        "player_season:obp",
        "player_season:slg",
    ]
    columns = [column for column in preferred if column in pivot.columns] or list(pivot.columns)[:12]
    pivot = pivot[columns]
    fig, ax = plt.subplots(figsize=(max(9, len(columns) * 0.8), max(4.8, len(pivot.index) * 0.45)))
    pivot.plot(kind="barh", ax=ax, width=0.82)
    ax.set_title("Available Rows By Method And Target")
    ax.set_xlabel("Available prediction rows")
    ax.set_ylabel("Method")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(title="Target", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def _plot_method_metric_matrix(metric_rows: list[dict[str, Any]], output_path: Path, *, metric_scope: str) -> str | None:
    if not metric_rows:
        return None
    try:
        import pandas as pd  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None
    df = pd.DataFrame(metric_rows)
    if df.empty:
        return None
    df = df[(df["metric_scope"] == metric_scope) & (df["prediction_level"].isin(["event", "player_season"]))].copy()
    df["primary_value"] = pd.to_numeric(df["primary_value"], errors="coerce")
    df = df.dropna(subset=["primary_value"])
    if df.empty:
        return None
    df["target_label"] = df["prediction_level"].astype(str) + ":" + df["target_name"].astype(str)
    preferred = [f"event:{target}" for target in EVENT_TARGET_ORDER] + [
        "player_season:avg_ev",
        "player_season:ba",
        "player_season:ops",
        "player_season:obp",
        "player_season:slg",
    ]
    pivot = df.pivot_table(index="label", columns="target_label", values="primary_value", aggfunc="first")
    columns = [column for column in preferred if column in pivot.columns] or list(pivot.columns)[:12]
    pivot = pivot[columns]
    if pivot.empty:
        return None
    fig, ax = plt.subplots(figsize=(max(9, len(columns) * 0.9), max(4.8, len(pivot.index) * 0.45)))
    matrix = pivot.astype(float).to_numpy()
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title(f"Primary Metric Matrix ({metric_scope})")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for y, _label in enumerate(pivot.index):
        for x, _target in enumerate(pivot.columns):
            value = pivot.iloc[y, x]
            if pd.notna(value):
                ax.text(x, y, f"{float(value):.3g}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def _plot_baseline_deltas(comparison_rows: list[dict[str, Any]], output_path: Path, *, baseline_key: str, metric_scope: str = "per_method_all_available") -> str | None:
    rows = [
        row
        for row in comparison_rows
        if row.get("baseline_method_key") == baseline_key
        and row.get("metric_scope") == metric_scope
        and row.get("prediction_level") == "event"
    ]
    if not rows:
        return None
    try:
        import pandas as pd  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None
    df = pd.DataFrame(rows)
    df["improvement_positive_is_candidate_better"] = pd.to_numeric(df["improvement_positive_is_candidate_better"], errors="coerce")
    df = df.dropna(subset=["improvement_positive_is_candidate_better"]).copy()
    df["target_order"] = df["target_name"].map({target: index for index, target in enumerate(EVENT_TARGET_ORDER)}).fillna(99)
    df = df.sort_values(["target_order", "candidate_label"]).head(60)
    if df.empty:
        return None
    labels = df["candidate_label"].astype(str) + " / " + df["target_name"].astype(str)
    colors = ["#0f766e" if value >= 0 else "#b91c1c" for value in df["improvement_positive_is_candidate_better"]]
    fig, ax = plt.subplots(figsize=(11, max(5, len(df) * 0.22)))
    ax.barh(labels, df["improvement_positive_is_candidate_better"], color=colors)
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_title(f"Improvement Versus {baseline_key} Baseline")
    ax.set_xlabel("Positive means candidate is better")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def _method_comparison_assets(
    base_dir: Path,
    run_profile: dict[str, Any],
    tables_dir: Path,
    figures_dir: Path,
) -> dict[str, Any]:
    root = _method_evaluation_root(base_dir, run_profile)
    source_tables = {
        "method_metrics": root / "tables" / "method_metrics.csv",
        "sample_counts": root / "tables" / "sample_counts.csv",
        "same_sample_intersection_metrics": root / "tables" / "same_sample_intersection_metrics.csv",
        "best_by_target": root / "tables" / "best_by_target.csv",
        "method_map": root / "tables" / "method_map.csv",
    }
    method_metrics = _read_csv_rows(source_tables["method_metrics"])
    sample_counts = _read_csv_rows(source_tables["sample_counts"])
    same_sample_metrics = _read_csv_rows(source_tables["same_sample_intersection_metrics"])
    best_by_target = _read_csv_rows(source_tables["best_by_target"])
    method_map = _read_csv_rows(source_tables["method_map"])

    combined_metrics = method_metrics + same_sample_metrics
    baseline_comparison = _baseline_comparison_rows(combined_metrics)

    output_tables = {
        "method_metrics": tables_dir / "method_evaluation_method_metrics.csv",
        "sample_counts": tables_dir / "method_evaluation_sample_counts.csv",
        "same_sample_intersection_metrics": tables_dir / "method_evaluation_same_sample_intersection_metrics.csv",
        "best_by_target": tables_dir / "method_evaluation_best_by_target.csv",
        "method_map": tables_dir / "method_evaluation_method_map.csv",
        "baseline_comparison": tables_dir / "method_baseline_comparison.csv",
    }
    _write_csv(output_tables["method_metrics"], method_metrics)
    _write_csv(output_tables["sample_counts"], sample_counts)
    _write_csv(output_tables["same_sample_intersection_metrics"], same_sample_metrics)
    _write_csv(output_tables["best_by_target"], best_by_target)
    _write_csv(output_tables["method_map"], method_map)
    _write_csv(output_tables["baseline_comparison"], baseline_comparison)

    figures = {
        "method_sample_counts": _plot_method_sample_counts(sample_counts, figures_dir / "method_available_rows_by_target.png"),
        "method_metric_matrix": _plot_method_metric_matrix(method_metrics, figures_dir / "method_primary_metric_matrix.png", metric_scope="per_method_all_available"),
        "same_sample_metric_matrix": _plot_method_metric_matrix(same_sample_metrics, figures_dir / "method_same_sample_metric_matrix.png", metric_scope="same_sample_intersection"),
        "delta_vs_context": _plot_baseline_deltas(baseline_comparison, figures_dir / "method_delta_vs_context.png", baseline_key="context"),
        "delta_vs_fusion": _plot_baseline_deltas(baseline_comparison, figures_dir / "method_delta_vs_fusion.png", baseline_key="fusion"),
    }
    return {
        "source_root": str(root),
        "source_tables": {key: str(path) for key, path in source_tables.items()},
        "output_tables": {key: str(path) for key, path in output_tables.items()},
        "figures": {key: value for key, value in figures.items() if value},
        "method_metrics_rows": len(method_metrics),
        "sample_count_rows": len(sample_counts),
        "same_sample_metric_rows": len(same_sample_metrics),
        "best_by_target_rows": len(best_by_target),
        "baseline_comparison_rows": baseline_comparison,
        "method_map_rows": method_map,
        "sample_count_rows_data": sample_counts,
        "method_metric_rows_data": method_metrics,
        "same_sample_metric_rows_data": same_sample_metrics,
        "best_by_target_rows_data": best_by_target,
    }


def _make_contact_sheet(base_dir: Path, full_run_id: str, output_path: Path, *, max_images: int = 12) -> str | None:
    frame_dir = base_dir / "debug" / full_run_id / "frames"
    if not frame_dir.exists():
        return None
    image_paths = sorted(frame_dir.glob("*.jpg"))[:max_images]
    if not image_paths:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception:
        return None
    thumbs = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((320, 180))
        canvas = Image.new("RGB", (320, 220), "white")
        canvas.paste(image, ((320 - image.width) // 2, 0))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        label = path.stem.replace("_contact", "")[:54]
        draw.text((8, 186), label, fill=(20, 27, 39), font=font)
        draw.text((8, 202), "Estimated contact frame", fill=(80, 88, 102), font=font)
        thumbs.append(canvas)
    cols = 3
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 320, rows * 220), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % cols) * 320, (index // cols) * 220))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return str(output_path)


def _write_methods_svg(output_path: Path) -> None:
    labels = [
        ("Statcast BBE\nUniverse", 40, 95),
        ("Official Video\nSources", 250, 95),
        ("Contact-Aligned\nClips", 460, 95),
        ("YOLO / Tracking\nPose / Bat Line", 670, 95),
        ("Sequence TCN\nVideoMAE\nContext CatBoost", 880, 95),
        ("Late Fusion\nStatcast Heads", 1090, 95),
    ]
    width = 1320
    height = 260
    blocks = []
    for text, x, y in labels:
        lines = text.split("\n")
        blocks.append(f'<rect x="{x}" y="{y}" width="160" height="76" rx="10" fill="#ffffff" stroke="#94a3b8" stroke-width="2"/>')
        for line_index, line in enumerate(lines):
            blocks.append(
                f'<text x="{x + 80}" y="{y + 30 + line_index * 20}" text-anchor="middle" '
                f'font-family="Arial, sans-serif" font-size="15" fill="#111827">{html_escape(line)}</text>'
            )
    arrows = []
    for _, x, y in labels[:-1]:
        arrows.append(
            f'<line x1="{x + 165}" y1="{y + 38}" x2="{x + 205}" y2="{y + 38}" stroke="#0f766e" stroke-width="3" marker-end="url(#arrow)"/>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">'
        '<path d="M0,0 L0,6 L9,3 z" fill="#0f766e"/></marker></defs>'
        '<rect width="100%" height="100%" fill="#f8fafc"/>'
        '<text x="40" y="42" font-family="Arial, sans-serif" font-size="26" font-weight="700" fill="#111827">'
        "Method Overview: Batting Video to Statcast-Style Targets</text>"
        + "".join(blocks + arrows)
        + "</svg>"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def _maybe_render_overlay_videos(
    base_dir: Path,
    full_run_id: str,
    videos_dir: Path,
    *,
    make_overlay_videos: bool,
    max_overlay_clips: int,
) -> dict[str, Any]:
    overlay_manifest = base_dir / "debug" / full_run_id / "cv_overlay_videos_v1.parquet"
    result: dict[str, Any] = {
        "existing_overlay_manifest": str(overlay_manifest),
        "existing_overlay_manifest_exists": overlay_manifest.exists(),
        "generated": False,
        "rows": _maybe_num_rows(overlay_manifest),
    }
    if not make_overlay_videos:
        return result
    from sport_pipeline.cv.overlay_video import render_cv_overlay_videos

    summary = render_cv_overlay_videos(
        base_dir,
        full_run_id,
        output_dir=videos_dir / "cv_overlays",
        overlay_manifest_path=videos_dir / "cv_overlay_videos_v1.parquet",
        debug_overlays_path=videos_dir / "debug_overlays_v1.parquet",
        summary_path=videos_dir / "cv_overlay_videos_summary.json",
        progress_path=videos_dir / "cv_overlay_videos_progress.json",
        max_clips=max_overlay_clips,
        overwrite=False,
        update_clips_manifest=False,
    )
    result.update(
        {
            "generated": True,
            "rendered_overlays": summary.get("rendered_overlays"),
            "selected_clips": summary.get("selected_clips"),
            "output_dir": summary.get("outputs", {}).get("overlay_dir"),
            "summary": summary,
        }
    )
    return result


def _relative_link(root: Path, target: str | Path | None) -> str:
    if not target:
        return ""
    path = Path(str(target))
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _render_figure_gallery(root: Path, figure_paths: Iterable[Path]) -> str:
    items = []
    for path in figure_paths:
        if not path.exists():
            continue
        rel = _relative_link(root, path)
        title = path.stem.replace("_", " ").title()
        items.append(
            '<article class="case-card">'
            f"<h3>{html_escape(title)}</h3>"
            f'<a href="{html_escape(rel)}"><img src="{html_escape(rel)}" alt="{html_escape(title)}" style="max-width:100%;height:auto;border:1px solid #d8dee8;border-radius:6px;"></a>'
            "</article>"
        )
    if not items:
        return "<p>No figures generated.</p>"
    return '<div class="case-grid">' + "".join(items) + "</div>"


def _render_video_links(root: Path, videos_dir: Path) -> str:
    videos = sorted(videos_dir.glob("**/*.mp4"))
    if not videos:
        return "<p>No overlay videos generated in this research output bundle yet.</p>"
    rows = [{"video": video.name, "path": _relative_link(root, video)} for video in videos[:50]]
    return render_table(("video", "path"), rows)


def build_research_outputs(
    *,
    base_dir: str | Path,
    run_profile: dict[str, Any],
    final_run_id: str | None = None,
    full_run_id: str | None = None,
    output_root: str | Path | None = None,
    make_overlay_videos: bool = False,
    max_overlay_clips: int = 12,
    max_prediction_plot_rows: int = 60000,
) -> dict[str, Any]:
    """Build paper-style figures, tables, an index page, and optional overlays."""

    base = Path(base_dir)
    resolved_full_run_id = full_run_id or profile_run_id(run_profile, "full_run_id", "mlb_2024_2026_full_v2")
    resolved_final_run_id = final_run_id or profile_run_id(run_profile, "fusion_run_id", "fusion_mlb_2024_2026_v2")
    report_root = Path(output_root) if output_root is not None else base / "reports"
    paths = _paths(report_root, resolved_final_run_id)
    _ensure_dirs(paths)

    run_ids = report_run_candidates(run_profile, include_smoke=False)
    metrics_rows = _collect_metrics(base, run_ids)
    artifact_rows = _artifact_rows(base, run_profile, resolved_full_run_id, resolved_final_run_id)
    availability_rows = _target_availability_rows(metrics_rows, resolved_final_run_id)
    model_design_rows = _model_design_rows(run_profile)
    fusion_source_status = _fusion_source_status_rows(base, resolved_final_run_id)
    provenance_checks = _provenance_checks(
        base,
        run_profile,
        resolved_full_run_id,
        resolved_final_run_id,
        fusion_source_status,
    )
    method_comparison = _method_comparison_assets(base, run_profile, paths.tables, paths.figures)

    _write_csv(paths.tables / "metrics_by_run_target.csv", metrics_rows)
    _write_csv(paths.tables / "artifact_inventory.csv", artifact_rows)
    _write_csv(paths.tables / "target_availability.csv", availability_rows)
    _write_csv(paths.tables / "model_design.csv", model_design_rows)
    _write_csv(paths.tables / "fusion_source_status.csv", fusion_source_status)
    _write_csv(paths.tables / "provenance_checks.csv", provenance_checks["checks"])

    figure_paths = [
        paths.figures / "method_overview.svg",
        paths.figures / "model_metrics_mae.png",
        paths.figures / "target_availability.png",
        paths.figures / "artifact_scale.png",
        paths.figures / "fusion_input_provenance.png",
        paths.figures / "prediction_scatter_ev_la.png",
        paths.figures / "residual_distribution_ev_la.png",
        paths.figures / "method_available_rows_by_target.png",
        paths.figures / "method_primary_metric_matrix.png",
        paths.figures / "method_same_sample_metric_matrix.png",
        paths.figures / "method_delta_vs_context.png",
        paths.figures / "method_delta_vs_fusion.png",
        paths.figures / "contact_frame_sheet.jpg",
    ]

    _write_methods_svg(paths.figures / "method_overview.svg")
    _plot_model_metrics(metrics_rows, paths.figures / "model_metrics_mae.png")
    _plot_target_availability(availability_rows, paths.figures / "target_availability.png")
    _plot_artifact_counts(artifact_rows, paths.figures / "artifact_scale.png")
    fusion_rows = _plot_fusion_audit(
        base,
        resolved_final_run_id,
        paths.figures / "fusion_input_provenance.png",
        paths.tables / "fusion_input_audit_summary.csv",
    )
    prediction_figures = _plot_predictions(
        base,
        resolved_final_run_id,
        paths.figures,
        paths.tables,
        max_rows=max_prediction_plot_rows,
    )
    contact_sheet = _make_contact_sheet(base, resolved_full_run_id, paths.figures / "contact_frame_sheet.jpg")
    overlay_summary = _maybe_render_overlay_videos(
        base,
        resolved_full_run_id,
        paths.videos,
        make_overlay_videos=make_overlay_videos,
        max_overlay_clips=max_overlay_clips,
    )

    summary = {
        "schema_version": "research_outputs_v1",
        "base_dir": str(base),
        "final_run_id": resolved_final_run_id,
        "full_run_id": resolved_full_run_id,
        "metrics_rows": len(metrics_rows),
        "artifact_rows": artifact_rows,
        "model_design_rows": model_design_rows,
        "target_availability": availability_rows,
        "fusion_source_status": fusion_source_status,
        "provenance_checks": provenance_checks,
        "method_comparison": {
            "source_root": method_comparison["source_root"],
            "source_tables": method_comparison["source_tables"],
            "output_tables": method_comparison["output_tables"],
            "figures": method_comparison["figures"],
            "method_metrics_rows": method_comparison["method_metrics_rows"],
            "sample_count_rows": method_comparison["sample_count_rows"],
            "same_sample_metric_rows": method_comparison["same_sample_metric_rows"],
            "best_by_target_rows": method_comparison["best_by_target_rows"],
            "baseline_comparison_rows": len(method_comparison["baseline_comparison_rows"]),
        },
        "fusion_input_summary_rows": fusion_rows,
        "prediction_figures": prediction_figures,
        "contact_sheet": contact_sheet,
        "overlay_summary": overlay_summary,
        "outputs": {
            "index_html": str(paths.index_html),
            "summary_json": str(paths.summary_json),
            "figures": str(paths.figures),
            "tables": str(paths.tables),
            "videos": str(paths.videos),
        },
    }
    _write_json(paths.summary_json, summary)

    metadata = {
        "base_dir": str(base),
        "final_run_id": resolved_final_run_id,
        "full_run_id": resolved_full_run_id,
        "figures_dir": str(paths.figures),
        "tables_dir": str(paths.tables),
        "videos_dir": str(paths.videos),
        "overlay_generation": "enabled" if make_overlay_videos else "disabled",
        "provenance_status": provenance_checks["overall_status"],
    }
    sections = (
        ("Research Output Bundle", render_kv_table(metadata)),
        ("Run Provenance And V2 Checks", render_table(("check", "status", "value", "expected"), provenance_checks["checks"])),
        ("Model Design Summary", render_table(("method", "run_id", "level", "inputs", "model", "targets", "notes"), model_design_rows)),
        ("Fusion Source Status", render_table(("run_id", "exists", "rows", "path"), fusion_source_status)),
        ("Method Map", render_table(("method_key", "label", "run_id", "method_family", "input_signal", "aggregation_scope", "player_season_projection_run_id", "what_it_tests_ja"), method_comparison["method_map_rows"])),
        ("Method Sample Counts", render_table(("label", "prediction_level", "target_name", "prediction_rows", "available_rows", "unique_samples"), method_comparison["sample_count_rows_data"])),
        ("Method Metrics By Target", render_table(("metric_scope", "label", "prediction_level", "target_name", "primary_metric", "primary_value", "n_available", "mae", "rmse", "r2", "spearman", "f1", "brier"), method_comparison["method_metric_rows_data"][:160])),
        ("Same-Sample Method Metrics", render_table(("label", "prediction_level", "target_name", "primary_metric", "primary_value", "intersection_samples", "n_available", "mae", "rmse", "r2", "spearman", "f1", "brier"), method_comparison["same_sample_metric_rows_data"][:160])),
        ("Each Method Versus Baselines", render_table(("metric_scope", "prediction_level", "target_name", "baseline_label", "candidate_label", "primary_metric", "baseline_value", "candidate_value", "improvement_positive_is_candidate_better", "baseline_n_available", "candidate_n_available"), method_comparison["baseline_comparison_rows"][:220])),
        ("Best Method By Target", render_table(("metric_scope", "prediction_level", "target_name", "best_label", "method_family", "primary_metric", "primary_value", "n_available"), method_comparison["best_by_target_rows_data"])),
        ("Figures", _render_figure_gallery(paths.root, figure_paths)),
        ("Target Availability", render_table(("target_name", "n_available", "n_skipped"), availability_rows)),
        ("Metric Table Preview", render_table(("run_id", "prediction_level", "target_name", "mae", "rmse", "r2", "brier", "f1", "n_available", "n_skipped"), metrics_rows[:80])),
        ("Artifact Inventory", render_table(("artifact", "exists", "rows_or_files", "path"), artifact_rows)),
        ("Overlay Video Links", _render_video_links(paths.root, paths.videos)),
    )
    html = render_page(
        "Research Output Bundle",
        resolved_final_run_id,
        sections,
        subtitle="Paper-ready figures, tables, and visual evidence generated from the Colab result artifacts.",
    )
    write_page(paths.index_html, html)
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    from sport_pipeline.pipeline.run_profile import load_run_profile

    parser = argparse.ArgumentParser(description="Build paper-style research outputs from full-run artifacts.")
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/baseball_vision")
    parser.add_argument("--run-profile", default=None)
    parser.add_argument("--final-run-id", default=None)
    parser.add_argument("--full-run-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--make-overlay-videos", action="store_true")
    parser.add_argument("--max-overlay-clips", type=int, default=12)
    parser.add_argument("--max-prediction-plot-rows", type=int, default=60000)
    args = parser.parse_args(argv)

    profile = load_run_profile(args.run_profile)
    summary = build_research_outputs(
        base_dir=args.base_dir,
        run_profile=profile,
        final_run_id=args.final_run_id,
        full_run_id=args.full_run_id,
        output_root=args.output_root,
        make_overlay_videos=args.make_overlay_videos,
        max_overlay_clips=args.max_overlay_clips,
        max_prediction_plot_rows=args.max_prediction_plot_rows,
    )
    print(json.dumps(summary["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
