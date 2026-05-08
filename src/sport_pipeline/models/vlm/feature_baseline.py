"""VLM mechanics feature template and lightweight baseline.

This module does not call a remote VLM or download large model weights. It
creates a table that a VLM step can fill with captions/tags/scores, then turns
those VLM outputs into dependency-light prediction rows.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from sport_pipeline.artifact_check import write_json
from sport_pipeline.evaluation import evaluate_predictions, load_target_registry, validate_prediction_rows
from sport_pipeline.io import read_table, write_table


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TARGET_REGISTRY = PROJECT_ROOT / "configs/targets/target_registry_v1.yaml"
EVENT_TARGETS = ("ev", "la", "hard_hit", "barrel", "xba", "xwoba")
VLM_KEYWORDS = (
    "leg kick",
    "load",
    "stride",
    "closed stance",
    "open stance",
    "balanced",
    "off balance",
    "early",
    "late",
    "contact",
    "follow through",
    "uppercut",
    "level swing",
    "hands",
    "hip rotation",
)
JSON_OBJECT_COLUMNS = ("vlm_labels", "numeric_features")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _to_float(value: Any) -> float | None:
    if _is_missing(value) or isinstance(value, bool):
        return None
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(output) or math.isinf(output):
        return None
    return output


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _serialise_json_object_columns(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    serialised = []
    for row in rows:
        output = dict(row)
        for column in JSON_OBJECT_COLUMNS:
            output[column] = json.dumps(_json_object(output.get(column)), ensure_ascii=False, sort_keys=True)
        serialised.append(output)
    return serialised


def _read_vlm_feature_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_table(path)
    for row in rows:
        for column in JSON_OBJECT_COLUMNS:
            row[column] = _json_object(row.get(column))
    return rows


def _write_vlm_feature_rows(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    return write_table(path, _serialise_json_object_columns(rows))


def _split_map(base_dir: Path) -> dict[str, str]:
    for relative in (
        "manifests/splits/player_group_split_v1.parquet",
        "manifests/splits/temporal_split_v1.parquet",
        "manifests/splits/player_group_split_v1.jsonl",
        "manifests/splits/temporal_split_v1.jsonl",
    ):
        path = base_dir / relative
        if path.exists():
            return {str(row["event_id"]): str(row.get("split", "unknown")) for row in read_table(path)}
    return {}


def _select_clips(clip_rows: Iterable[dict[str, Any]], max_clips: int | None = None) -> list[dict[str, Any]]:
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clip_rows:
        if row.get("clip_status") == "excluded":
            continue
        by_event[str(row.get("event_id"))].append(row)
    selected = []
    for rows in by_event.values():
        selected.append(
            sorted(
                rows,
                key=lambda row: (
                    1.0 if row.get("clip_status") == "clean_clip" else 0.0,
                    1.0 if row.get("quality_tier") == "usable_primary" else 0.0,
                    _to_float(row.get("contact_confidence")) or 0.0,
                    str(row.get("clip_id", "")),
                ),
                reverse=True,
            )[0]
        )
    selected = sorted(selected, key=lambda row: str(row.get("event_id", "")))
    return selected[:max_clips] if max_clips is not None else selected


def _prompt_for_clip(row: dict[str, Any]) -> str:
    return (
        "Analyze the batter's mechanics in this contact-centered baseball swing clip. "
        "Return concise JSON with: stance, load, stride, bat_path, contact_timing, balance, "
        "hip_rotation_score, bat_path_steepness_score, timing_score, and a one sentence mechanics_caption. "
        "Do not infer Statcast labels directly."
    )


def build_vlm_feature_template(
    base_dir: str | Path,
    *,
    clip_run_id: str = "mlb_2024_2026_full_v1",
    vlm_feature_id: str = "vlm_mechanics_v1",
    clips_path: str | Path | None = None,
    max_clips: int | None = None,
    output_suffix: str = ".parquet",
) -> dict[str, Path]:
    """Write a VLM input template under ``features/{vlm_feature_id}``."""

    base = Path(base_dir)
    clips_file = Path(clips_path) if clips_path else base / f"clips/{clip_run_id}/clips_v1.parquet"
    clip_rows = read_table(clips_file) if clips_file.exists() else []
    splits = _split_map(base)
    selected = _select_clips(clip_rows, max_clips=max_clips)
    outputs = {
        "template": base / f"features/{vlm_feature_id}/manifest{output_suffix}",
        "summary": base / f"reports/preflight/vlm_feature_template_{vlm_feature_id}.json",
    }
    existing_rows = _read_vlm_feature_rows(outputs["template"]) if outputs["template"].exists() else []
    existing_by_clip_id = {
        str(row.get("clip_id") or row.get("sample_id")): row
        for row in existing_rows
        if row.get("clip_id") is not None or row.get("sample_id") is not None
    }
    rows = []
    preserved_rows = 0
    for clip in selected:
        row = {
            "schema_version": "vlm_mechanics_features_v1",
            "sample_id": str(clip.get("clip_id")),
            "clip_id": str(clip.get("clip_id")),
            "event_id": str(clip.get("event_id")),
            "same_event_group_id": str(clip.get("same_event_group_id") or clip.get("event_id")),
            "batter_id": clip.get("batter_id"),
            "season": clip.get("season"),
            "batter_season_id": str(clip.get("batter_season_id")),
            "split": splits.get(str(clip.get("event_id")), str(clip.get("split") or "unknown")),
            "clip_path": clip.get("clip_path"),
            "debug_frame_path": clip.get("debug_frame_path"),
            "view_label": clip.get("view_label"),
            "contact_frame": clip.get("contact_frame"),
            "contact_confidence": clip.get("contact_confidence"),
            "view_confidence": clip.get("view_confidence"),
            "vlm_prompt": _prompt_for_clip(clip),
            "vlm_model": None,
            "vlm_caption": None,
            "vlm_labels": {},
            "numeric_features": {},
            "feature_status": "needs_vlm",
        }
        existing = existing_by_clip_id.get(str(row["clip_id"]))
        if existing is not None:
            preserved = False
            for key in ("vlm_model", "vlm_caption", "vlm_labels", "numeric_features", "vlm_raw_response", "vlm_error", "feature_status"):
                if key in existing and not _is_missing(existing.get(key)):
                    row[key] = existing.get(key)
                    preserved = True
            if preserved:
                preserved_rows += 1
        rows.append(row)
    _write_vlm_feature_rows(outputs["template"], rows)
    write_json(
        {
            "schema_version": "vlm_feature_template_summary_v1",
            "clip_run_id": clip_run_id,
            "vlm_feature_id": vlm_feature_id,
            "clips_path": str(clips_file),
            "input_clips": len(clip_rows),
            "existing_template_rows": len(existing_rows),
            "preserved_existing_vlm_rows": preserved_rows,
            "template_rows": len(rows),
            "note_ja": "この manifest に VLM caption / labels / numeric_features を追加してから 24 の baseline cell を実行する。",
            "outputs": {key: str(path) for key, path in outputs.items()},
        },
        outputs["summary"],
    )
    return outputs


def _event_map(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["event_id"]): row for row in rows if row.get("event_id") is not None}


def _numeric_values(value: Any) -> list[float]:
    values: list[float] = []
    if isinstance(value, dict):
        for item in value.values():
            converted = _to_float(item)
            if converted is not None:
                values.append(converted)
    elif isinstance(value, list):
        for item in value:
            converted = _to_float(item)
            if converted is not None:
                values.append(converted)
    return values


def _caption_features(text: str) -> list[float]:
    lower = text.lower()
    length_feature = min(len(lower) / 500.0, 1.0)
    keyword_features = [1.0 if keyword in lower else 0.0 for keyword in VLM_KEYWORDS]
    return [length_feature] + keyword_features


def _vlm_vector(row: dict[str, Any]) -> list[float]:
    caption = str(row.get("vlm_caption") or row.get("mechanics_caption") or "")
    numeric = _numeric_values(row.get("numeric_features")) + _numeric_values(row.get("vlm_labels"))
    if not caption and not numeric:
        return []
    vector = _caption_features(caption)
    vector.extend(numeric[:24])
    if not numeric:
        vector.extend([0.0, 0.0, 0.0])
    return [float(value) for value in vector]


def _fit_univariate(samples: list[dict[str, Any]], target_column: str) -> dict[str, float] | None:
    train_rows = [row for row in samples if row.get("split") == "train" and not _is_missing(row.get(target_column))]
    if not train_rows:
        train_rows = [row for row in samples if not _is_missing(row.get(target_column))]
    if not train_rows:
        return None
    xs = [float(row["vlm_signal"]) for row in train_rows]
    ys = [float(row[target_column]) for row in train_rows]
    mean_x = mean(xs)
    mean_y = mean(ys)
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    slope = 0.0 if variance_x == 0 else sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / variance_x
    return {"intercept": float(mean_y - slope * mean_x), "slope": float(slope), "train_rows": float(len(train_rows))}


def _sample_from_vlm_row(row: dict[str, Any], event: dict[str, Any], vector: list[float]) -> dict[str, Any]:
    sample = {
        "sample_id": str(row.get("sample_id") or row.get("clip_id")),
        "clip_id": str(row.get("clip_id")),
        "event_id": str(row.get("event_id")),
        "same_event_group_id": str(row.get("same_event_group_id") or row.get("event_id")),
        "batter_season_id": str(row.get("batter_season_id") or event.get("batter_season_id")),
        "split": str(row.get("split") or event.get("split") or "unknown"),
        "vlm_model": row.get("vlm_model"),
        "vlm_signal": mean(vector) if vector else 0.0,
        "vlm_feature_dim": len(vector),
        "vlm_caption_present": bool(row.get("vlm_caption") or row.get("mechanics_caption")),
    }
    for column in (
        "launch_speed",
        "launch_angle",
        "target_hard_hit",
        "target_barrel",
        "estimated_ba_using_speedangle",
        "estimated_woba_using_speedangle",
    ):
        sample[column] = event.get(column)
    return sample


def run_vlm_feature_baseline(
    base_dir: str | Path,
    *,
    prediction_run_id: str = "vlm_mechanics_mlb_2024_2026_v1",
    vlm_feature_id: str = "vlm_mechanics_v1",
    vlm_features_path: str | Path | None = None,
    bbe_events: str | Path | None = None,
    target_registry: str | Path = DEFAULT_TARGET_REGISTRY,
    require_non_empty: bool = False,
    output_suffix: str = ".parquet",
) -> dict[str, Path]:
    """Train a lightweight event-level head from VLM caption/tag features."""

    base = Path(base_dir)
    feature_path = Path(vlm_features_path) if vlm_features_path else base / f"features/{vlm_feature_id}/manifest{output_suffix}"
    bbe_path = Path(bbe_events) if bbe_events else base / "manifests/bbe_events_v1.parquet"
    outputs = {
        "predictions": base / f"predictions/{prediction_run_id}/predictions_v1{output_suffix}",
        "metrics": base / f"predictions/{prediction_run_id}/metrics_v1.json",
        "summary": base / f"reports/preflight/vlm_feature_baseline_{prediction_run_id}.json",
        "feature_samples": base / f"datasets/vlm_feature_samples/{prediction_run_id}/manifest{output_suffix}",
    }
    targets = load_target_registry(target_registry)
    events = _event_map(read_table(bbe_path)) if bbe_path.exists() else {}
    feature_rows = _read_vlm_feature_rows(feature_path) if feature_path.exists() else []
    samples = []
    skipped = []
    for row in feature_rows:
        vector = _vlm_vector(row)
        if not vector:
            skipped.append({"clip_id": row.get("clip_id"), "reason": "missing_vlm_caption_or_numeric_features"})
            continue
        event = events.get(str(row.get("event_id")))
        if event is None:
            skipped.append({"clip_id": row.get("clip_id"), "reason": "event_not_found"})
            continue
        samples.append(_sample_from_vlm_row(row, event, vector))

    fitted = {
        target_name: _fit_univariate(samples, targets[target_name].column)
        for target_name in EVENT_TARGETS
        if target_name in targets
    }
    predictions = []
    for sample in samples:
        for target_name in EVENT_TARGETS:
            if target_name not in targets:
                continue
            target = targets[target_name]
            y_true = sample.get(target.column)
            model = fitted.get(target_name)
            available = not _is_missing(y_true) and model is not None
            y_pred = None
            missing_reason = None
            if available:
                y_pred = float(model["intercept"] + model["slope"] * float(sample["vlm_signal"]))
                if target.kind in {"binary", "probability"}:
                    y_pred = min(max(y_pred, 0.0), 1.0)
            elif _is_missing(y_true):
                missing_reason = "label_missing"
            else:
                missing_reason = "vlm_feature_baseline_not_fit_for_target"
            predictions.append(
                {
                    "run_id": prediction_run_id,
                    "sample_id": str(sample["sample_id"]),
                    "event_id": str(sample["event_id"]),
                    "batter_season_id": str(sample["batter_season_id"]),
                    "prediction_level": "event",
                    "target_name": target_name,
                    "y_true": None if _is_missing(y_true) else float(y_true),
                    "y_pred": y_pred,
                    "target_available": available,
                    "target_source": target.column,
                    "head_kind": target.kind,
                    "loss_name": target.loss,
                    "aggregation_scope": "vlm_mechanics_features",
                    "prior_mode": "none",
                    "label_missing_reason": missing_reason,
                    "requires_pa_manifest": target.requires_pa_manifest,
                    "n_prior_clips": 0,
                    "aggregation_method": "vlm_feature_univariate_head",
                    "same_event_ensemble": False,
                    "prediction_std": None,
                    "split": str(sample.get("split", "unknown")),
                }
            )

    if require_non_empty and not samples:
        write_json(
            {
                "schema_version": "vlm_feature_baseline_summary_v1",
                "prediction_run_id": prediction_run_id,
                "vlm_feature_id": vlm_feature_id,
                "error": "no_vlm_feature_samples",
                "feature_path": str(feature_path),
                "skipped": skipped[:100],
            },
            outputs["summary"],
        )
        raise RuntimeError("VLM feature baseline found 0 usable rows; fill vlm_caption, vlm_labels, or numeric_features first")

    validate_prediction_rows(predictions)
    metrics = evaluate_predictions(predictions, targets, run_id=prediction_run_id)
    write_table(outputs["feature_samples"], samples)
    write_table(outputs["predictions"], predictions)
    write_json(metrics, outputs["metrics"])
    write_json(
        {
            "schema_version": "vlm_feature_baseline_summary_v1",
            "prediction_run_id": prediction_run_id,
            "vlm_feature_id": vlm_feature_id,
            "feature_path": str(feature_path),
            "bbe_events": str(bbe_path),
            "feature_rows": len(feature_rows),
            "usable_samples": len(samples),
            "prediction_rows": len(predictions),
            "fitted_targets": {key: value for key, value in fitted.items() if value is not None},
            "skipped": skipped[:100],
            "note_ja": "VLM 自体はここでは実行しない。外部 VLM が書いた caption/tags/scores を prediction head に変換する段階。",
            "outputs": {key: str(path) for key, path in outputs.items() if key != "summary"},
        },
        outputs["summary"],
    )
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build VLM templates or run VLM feature baseline.")
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/baseball_vision")
    parser.add_argument("--mode", choices=("template", "baseline"), default="baseline")
    parser.add_argument("--clip-run-id", default="mlb_2024_2026_full_v1")
    parser.add_argument("--prediction-run-id", default="vlm_mechanics_mlb_2024_2026_v1")
    parser.add_argument("--vlm-feature-id", default="vlm_mechanics_v1")
    parser.add_argument("--vlm-features", default=None)
    parser.add_argument("--bbe-events", default=None)
    parser.add_argument("--clips", default=None)
    parser.add_argument("--target-registry", default=str(DEFAULT_TARGET_REGISTRY))
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--require-non-empty", action="store_true")
    parser.add_argument("--output-format", choices=("parquet", "jsonl", "json", "csv"), default="parquet")
    args = parser.parse_args(argv)
    if args.mode == "template":
        outputs = build_vlm_feature_template(
            args.base_dir,
            clip_run_id=args.clip_run_id,
            vlm_feature_id=args.vlm_feature_id,
            clips_path=args.clips,
            max_clips=args.max_clips,
            output_suffix="." + args.output_format,
        )
    else:
        outputs = run_vlm_feature_baseline(
            args.base_dir,
            prediction_run_id=args.prediction_run_id,
            vlm_feature_id=args.vlm_feature_id,
            vlm_features_path=args.vlm_features,
            bbe_events=args.bbe_events,
            target_registry=args.target_registry,
            require_non_empty=args.require_non_empty,
            output_suffix="." + args.output_format,
        )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
