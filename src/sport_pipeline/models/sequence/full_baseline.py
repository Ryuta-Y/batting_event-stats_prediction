"""Full structured-sequence baseline runner for Colab artifacts.

This module is deliberately dependency-light. It turns clean ``clips_v1`` rows
into structured sequence metadata, deterministic clip embeddings, player-season
prior rows, and predictions_v1 rows. Learned TCN/Transformer training can plug
into the same artifacts later without changing downstream contracts.
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
from sport_pipeline.datasets.sequence.contracts import EVENT_WITH_PLAYER_PRIOR_SCHEMA, SEQUENCE_DATASET_SCHEMA
from sport_pipeline.datasets.sequence.prior import build_event_with_prior_row, select_prior_clip_embeddings
from sport_pipeline.evaluation import evaluate_predictions, load_target_registry, validate_prediction_rows
from sport_pipeline.evaluation.target_registry import TargetSpec
from sport_pipeline.features import (
    CLIP_EMBEDDING_SCHEMA,
    PLAYER_SEASON_EMBEDDING_SCHEMA,
    STRUCTURED_FRAME_FEATURES_SCHEMA,
    STRUCTURED_SEQUENCE_MANIFEST_SCHEMA,
)
from sport_pipeline.features.embeddings import build_clip_embedding_row
from sport_pipeline.features.sequence_builder import (
    DEFAULT_FEATURE_NAMES,
    build_frame_feature_row,
    build_sequence_manifest_row,
)
from sport_pipeline.io import read_table, write_table
from sport_pipeline.models.sequence.aggregation import aggregate_clip_embeddings
from sport_pipeline.models.sequence.predictions import build_sequence_prediction_row
from sport_pipeline.schemas.data_manifest import validate_rows


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TARGET_REGISTRY = PROJECT_ROOT / "configs/targets/target_registry_v1.yaml"
EVENT_TARGETS = ("ev", "la", "hard_hit", "barrel", "xba", "xwoba")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _to_float(value: Any, default: float = 0.0) -> float:
    if _is_missing(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    if _is_missing(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_optional_table(path: Path) -> list[dict[str, Any]]:
    return read_table(path) if path.exists() else []


def _event_map(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["event_id"]): row for row in rows if row.get("event_id") is not None}


def _split_map(base_dir: Path) -> dict[str, str]:
    for relative in (
        "manifests/splits/temporal_split_v1.parquet",
        "manifests/splits/player_group_split_v1.parquet",
        "manifests/splits/temporal_split_v1.jsonl",
        "manifests/splits/player_group_split_v1.jsonl",
    ):
        path = base_dir / relative
        if path.exists():
            return {str(row["event_id"]): str(row.get("split", "unknown")) for row in read_table(path)}
    return {}


def _clip_score(row: dict[str, Any]) -> tuple[float, float, float, str]:
    clean_bonus = 1.0 if row.get("clip_status") == "clean_clip" else 0.0
    quality_bonus = 1.0 if row.get("quality_tier") == "usable_primary" else 0.0
    confidence = (
        _to_float(row.get("contact_confidence"))
        + _to_float(row.get("view_confidence"))
        + _to_float(row.get("batter_visibility_score"))
        + _to_float(row.get("bat_visibility_score"))
        + _to_float(row.get("plate_visibility_score"))
    )
    return (clean_bonus, quality_bonus, confidence, str(row.get("clip_id", "")))


def _select_representative_clean_clips(clip_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clip_rows:
        if row.get("clip_status") != "clean_clip":
            continue
        if row.get("quality_tier") != "usable_primary":
            continue
        if not row.get("clean_cohort_eligible"):
            continue
        if row.get("target_alignment_status") != "event_aligned":
            continue
        by_event[str(row["event_id"])].append(row)
    selected: list[dict[str, Any]] = []
    for rows in by_event.values():
        selected.append(sorted(rows, key=_clip_score, reverse=True)[0])
    return selected


def _enrich_clip(row: dict[str, Any], event: dict[str, Any], split: str) -> dict[str, Any]:
    enriched = dict(row)
    enriched["game_date"] = str(event.get("game_date", row.get("game_date", "1970-01-01")))
    enriched["split"] = split
    return enriched


def _phase_for_relative_time(relative_time: float | None) -> tuple[str, float]:
    if relative_time is None:
        return "unknown", 0.0
    if relative_time < -0.90:
        return "stance_load", 0.65
    if relative_time < -0.55:
        return "stride", 0.70
    if relative_time < -0.20:
        return "launch", 0.75
    if relative_time < -0.04:
        return "swing", 0.80
    if relative_time <= 0.08:
        return "contact", 0.90
    return "follow_through", 0.70


def _feature_values_for_frame(clip_row: dict[str, Any], frame_index: int, n_frames: int, relative_time: float | None) -> list[float]:
    progress = 0.0 if n_frames <= 1 else frame_index / (n_frames - 1)
    rel = 0.0 if relative_time is None else relative_time
    contact_strength = max(0.0, 1.0 - min(abs(rel) / 0.6, 1.0))
    view_conf = _to_float(clip_row.get("view_confidence"), 0.0)
    contact_conf = _to_float(clip_row.get("contact_confidence"), 0.0)
    batter_vis = _to_float(clip_row.get("batter_visibility_score"), 0.0)
    bat_vis = _to_float(clip_row.get("bat_visibility_score"), 0.0)
    plate_vis = _to_float(clip_row.get("plate_visibility_score"), 0.0)
    duration = max(_to_float(clip_row.get("duration_sec"), 1.0), 1e-6)
    return [
        0.45 + 0.10 * progress,
        0.58 - 0.03 * contact_strength,
        -18.0 + 42.0 * progress,
        contact_strength * contact_conf,
        max(0.0, contact_strength - 0.10) * contact_conf,
        -35.0 + 75.0 * progress,
        contact_strength / duration,
        bat_vis,
        plate_vis * (0.5 + 0.5 * view_conf),
        batter_vis,
    ]


def _build_sequence_rows(
    clip_rows: list[dict[str, Any]],
    frame_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sequence_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for clip in clip_rows:
        n_frames = max(2, int(frame_count))
        sequence = build_sequence_manifest_row(
            clip,
            n_frames=n_frames,
            feature_names=DEFAULT_FEATURE_NAMES,
            split=str(clip.get("split", "unknown")),
            sequence_path=None,
            target_available=True,
            target_missing_reason=None,
        )
        sequence_rows.append(sequence)
        start_time = _to_float(clip.get("start_time_sec"), 0.0)
        end_time = _to_float(clip.get("end_time_sec"), start_time + _to_float(clip.get("duration_sec"), 1.0))
        contact_time = None if _is_missing(clip.get("contact_time_sec")) else _to_float(clip.get("contact_time_sec"))
        for frame_index in range(n_frames):
            progress = 0.0 if n_frames <= 1 else frame_index / (n_frames - 1)
            time_sec = start_time + (end_time - start_time) * progress
            relative_time = None if contact_time is None else time_sec - contact_time
            phase, phase_conf = _phase_for_relative_time(relative_time)
            frame_rows.append(
                build_frame_feature_row(
                    sequence,
                    frame_index=frame_index,
                    time_sec=time_sec,
                    relative_time_to_contact_sec=relative_time,
                    phase_label=phase,
                    phase_confidence=phase_conf,
                    feature_values=_feature_values_for_frame(clip, frame_index, n_frames, relative_time),
                    feature_mask=[True] * len(DEFAULT_FEATURE_NAMES),
                )
            )
    return sequence_rows, frame_rows


def _embedding_from_sequence(sequence: dict[str, Any], frames: list[dict[str, Any]]) -> list[float]:
    values_by_index = list(zip(*[frame["feature_values"] for frame in frames]))
    means = [mean(values) for values in values_by_index]
    contact_rate = mean(1.0 if frame["phase_label"] == "contact" else 0.0 for frame in frames)
    return [
        means[0],
        means[2] / 90.0,
        means[3],
        means[6],
        _to_float(sequence.get("frame_rate"), 0.0) / 60.0,
        1.0 if sequence.get("clip_status") == "clean_clip" else 0.0,
        _to_float(sequence.get("n_frames"), 0.0) / 32.0,
        contact_rate,
    ]


def _build_clip_embedding_rows(
    sequence_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    clip_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    frames_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frame_rows:
        frames_by_sequence[str(frame["sequence_id"])].append(frame)
    rows: list[dict[str, Any]] = []
    for sequence in sequence_rows:
        clip = clip_by_id[str(sequence["clip_id"])]
        rows.append(
            build_clip_embedding_row(
                sequence,
                _embedding_from_sequence(sequence, frames_by_sequence[str(sequence["sequence_id"])]),
                encoder_name="structured_sequence_deterministic_encoder",
                encoder_version="full_baseline_v1",
                embedding_path=None,
                contact_confidence=_to_float(clip.get("contact_confidence"), 0.0),
                view_confidence=_to_float(clip.get("view_confidence"), 0.0),
                pose_coverage=_to_float(clip.get("batter_visibility_score"), 0.0),
                clean_cohort_eligible=bool(clip.get("clean_cohort_eligible")),
                target_alignment_status=str(clip.get("target_alignment_status", "event_aligned")),
            )
        )
    return rows


def _sequence_dataset_row(sequence: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    target_available = bool(
        event.get("target_ev_available", event.get("launch_speed") is not None)
        and event.get("target_la_available", event.get("launch_angle") is not None)
    )
    return {
        "schema_version": "sequence_dataset_contract_v1",
        "sample_id": sequence["sample_id"],
        "sequence_id": sequence["sequence_id"],
        "clip_id": sequence["clip_id"],
        "event_id": sequence["event_id"],
        "same_event_group_id": sequence["same_event_group_id"],
        "batter_id": sequence["batter_id"],
        "season": sequence["season"],
        "batter_season_id": sequence["batter_season_id"],
        "game_date": sequence["game_date"],
        "split": sequence["split"],
        "sequence_path": sequence.get("sequence_path"),
        "context_feature_path": None,
        "target_registry_version": "target_registry_v1",
        "prediction_level": "event",
        "aggregation_scope": "current_event_only",
        "prior_mode": "none",
        "n_prior_clips": 0,
        "quality_tier": sequence["quality_tier"],
        "view_label": sequence["view_label"],
        "target_available": target_available,
        "label_missing_reason": None if target_available else event.get("label_missing_reason", "event_target_missing"),
    }


def _fit_target_means(events: Iterable[dict[str, Any]], targets: dict[str, TargetSpec], train_event_ids: set[str]) -> dict[str, float]:
    means: dict[str, float] = {}
    event_list = list(events)
    for target_name in EVENT_TARGETS:
        target = targets[target_name]
        values: list[float] = []
        for event in event_list:
            if train_event_ids and str(event.get("event_id")) not in train_event_ids:
                continue
            value = event.get(target.column)
            if not _is_missing(value):
                values.append(float(value))
        if not values and train_event_ids:
            for event in event_list:
                value = event.get(target.column)
                if not _is_missing(value):
                    values.append(float(value))
        if values:
            means[target_name] = mean(values)
    return means


def _predict_from_event_prior(
    event_prior: dict[str, Any],
    event: dict[str, Any],
    targets: dict[str, TargetSpec],
    target_means: dict[str, float],
    run_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_values = event_prior.get("current_clip_embedding_values") or []
    prior_values = event_prior.get("player_season_embedding_values") or []
    current_signal = mean(current_values) if current_values else 0.0
    prior_signal = mean(prior_values) if prior_values else 0.0
    small_adjustment = 0.02 * current_signal + 0.01 * prior_signal
    for target_name in EVENT_TARGETS:
        target = targets[target_name]
        y_true = event.get(target.column)
        target_available = not _is_missing(y_true) and target_name in target_means
        missing_reason = None
        y_pred = None
        if target_available:
            base = target_means[target_name]
            y_pred = base + small_adjustment
            if target.kind in {"binary", "probability"}:
                y_pred = min(max(y_pred, 0.0), 1.0)
        elif _is_missing(y_true):
            missing_reason = event.get(f"target_{target_name}_missing_reason") or event.get("label_missing_reason") or "label_missing"
        else:
            missing_reason = "sequence_baseline_not_fit_for_target"
        rows.append(
            build_sequence_prediction_row(
                run_id,
                event_prior,
                target_name=target_name,
                y_true=None if _is_missing(y_true) else float(y_true),
                y_pred=None if y_pred is None else float(y_pred),
                target_available=target_available,
                target_source=target.column,
                head_kind=target.kind,
                loss_name=target.loss,
                label_missing_reason=missing_reason,
                requires_pa_manifest=target.requires_pa_manifest,
                prediction_std=None,
            )
        )
    return rows


def run_full_sequence_baseline(
    base_dir: str | Path,
    *,
    clip_run_id: str = "mlb_2024_2026_full_v1",
    prediction_run_id: str = "sequence_structured_mlb_2024_2026_v1",
    bbe_events: str | Path | None = None,
    clips_path: str | Path | None = None,
    target_registry: str | Path = DEFAULT_TARGET_REGISTRY,
    prior_mode: str = "past_only",
    aggregation_method: str = "quality_weighted_pooling",
    frame_count: int = 32,
    require_non_empty: bool = False,
    output_suffix: str = ".parquet",
    structured_sequence_feature_id: str = "structured_sequence_v1",
    clip_embedding_feature_id: str = "clip_embedding_v1",
    player_season_embedding_feature_id: str = "player_season_embedding_v1",
    sequence_dataset_id: str = "sequence_dataset_v1",
    event_with_prior_dataset_id: str = "event_with_player_prior_v1",
) -> dict[str, Path]:
    """Build sequence artifacts, player-season prior rows, predictions, and metrics."""

    base = Path(base_dir)
    bbe_path = Path(bbe_events) if bbe_events else base / "manifests/bbe_events_v1.parquet"
    clips = Path(clips_path) if clips_path else base / f"clips/{clip_run_id}/clips_v1.parquet"
    bbe_rows = read_table(bbe_path)
    clip_rows_raw = _read_optional_table(clips)
    events = _event_map(bbe_rows)
    splits = _split_map(base)
    selected_clips: list[dict[str, Any]] = []
    for clip in _select_representative_clean_clips(clip_rows_raw):
        event = events.get(str(clip["event_id"]))
        if event is None:
            continue
        selected_clips.append(_enrich_clip(clip, event, splits.get(str(clip["event_id"]), "unknown")))

    sequence_rows, frame_rows = _build_sequence_rows(selected_clips, frame_count=frame_count)
    clip_by_id = {str(clip["clip_id"]): clip for clip in selected_clips}
    clip_embeddings = _build_clip_embedding_rows(sequence_rows, frame_rows, clip_by_id) if sequence_rows else []
    embedding_by_clip = {str(row["clip_id"]): row for row in clip_embeddings}

    sequence_dataset_rows = [
        _sequence_dataset_row(sequence, events[str(sequence["event_id"])])
        for sequence in sequence_rows
        if str(sequence["event_id"]) in events
    ]

    event_prior_rows: list[dict[str, Any]] = []
    player_prior_rows: list[dict[str, Any]] = []
    for sequence in sequence_rows:
        event = dict(events[str(sequence["event_id"])])
        event["split"] = sequence["split"]
        current_clip = embedding_by_clip[str(sequence["clip_id"])]
        prior_row = build_event_with_prior_row(
            event,
            current_clip,
            clip_embeddings,
            prior_mode=prior_mode,
            aggregation_method=aggregation_method,
        )
        event_prior_rows.append(prior_row)
        selected = select_prior_clip_embeddings(event, clip_embeddings, prior_mode)
        if selected:
            player_prior_rows.append(aggregate_clip_embeddings(selected, aggregation_method, event, prior_mode))

    targets = load_target_registry(target_registry)
    train_event_ids = {
        str(row["event_id"])
        for row in sequence_dataset_rows
        if row.get("split") == "train"
    }
    target_means = _fit_target_means(
        (events[str(row["event_id"])] for row in sequence_dataset_rows if str(row["event_id"]) in events),
        targets,
        train_event_ids,
    )
    predictions: list[dict[str, Any]] = []
    for event_prior in event_prior_rows:
        event = events[str(event_prior["event_id"])]
        predictions.extend(_predict_from_event_prior(event_prior, event, targets, target_means, prediction_run_id))

    outputs = {
        "structured_sequence_manifest": base / f"features/{structured_sequence_feature_id}/manifest{output_suffix}",
        "structured_sequence_frames": base / f"features/{structured_sequence_feature_id}/frames{output_suffix}",
        "clip_embeddings": base / f"features/{clip_embedding_feature_id}/manifest{output_suffix}",
        "player_season_embeddings": base / f"features/{player_season_embedding_feature_id}/manifest{output_suffix}",
        "sequence_dataset": base / f"datasets/{sequence_dataset_id}/manifest{output_suffix}",
        "event_with_player_prior": base / f"datasets/{event_with_prior_dataset_id}/manifest{output_suffix}",
        "predictions": base / f"predictions/{prediction_run_id}/predictions_v1{output_suffix}",
        "metrics": base / f"predictions/{prediction_run_id}/metrics_v1.json",
        "summary": base / f"reports/preflight/full_sequence_baseline_{prediction_run_id}.json",
    }
    summary_payload = {
        "schema_version": "full_sequence_baseline_summary_v1",
        "clip_run_id": clip_run_id,
        "prediction_run_id": prediction_run_id,
        "artifact_namespace": {
            "structured_sequence_feature_id": structured_sequence_feature_id,
            "clip_embedding_feature_id": clip_embedding_feature_id,
            "player_season_embedding_feature_id": player_season_embedding_feature_id,
            "sequence_dataset_id": sequence_dataset_id,
            "event_with_prior_dataset_id": event_with_prior_dataset_id,
        },
        "prior_mode": prior_mode,
        "aggregation_method": aggregation_method,
        "require_non_empty": require_non_empty,
        "input_clips": len(clip_rows_raw),
        "selected_clean_events": len(selected_clips),
        "sequence_rows": len(sequence_rows),
        "frame_rows": len(frame_rows),
        "clip_embeddings": len(clip_embeddings),
        "player_season_embeddings": len(player_prior_rows),
        "event_with_prior_rows": len(event_prior_rows),
        "prediction_rows": len(predictions),
        "target_means_available": sorted(target_means),
    }
    if require_non_empty and not sequence_rows:
        write_json(summary_payload, outputs["summary"])
        raise RuntimeError(
            "full sequence baseline produced 0 sequence rows; not writing empty sequence/prediction artifacts "
            "in full-run mode. Check that 12 produced non-empty clips_v1 with clean_clip / usable_primary rows. "
            f"summary_path={outputs['summary']}"
        )

    validate_rows(STRUCTURED_SEQUENCE_MANIFEST_SCHEMA, sequence_rows)
    validate_rows(STRUCTURED_FRAME_FEATURES_SCHEMA, frame_rows)
    validate_rows(CLIP_EMBEDDING_SCHEMA, clip_embeddings)
    validate_rows(PLAYER_SEASON_EMBEDDING_SCHEMA, player_prior_rows)
    validate_rows(SEQUENCE_DATASET_SCHEMA, sequence_dataset_rows)
    validate_rows(EVENT_WITH_PLAYER_PRIOR_SCHEMA, event_prior_rows)
    validate_prediction_rows(predictions)
    metrics = evaluate_predictions(predictions, targets, run_id=prediction_run_id)
    write_table(outputs["structured_sequence_manifest"], sequence_rows)
    write_table(outputs["structured_sequence_frames"], frame_rows)
    write_table(outputs["clip_embeddings"], clip_embeddings)
    write_table(outputs["player_season_embeddings"], player_prior_rows)
    write_table(outputs["sequence_dataset"], sequence_dataset_rows)
    write_table(outputs["event_with_player_prior"], event_prior_rows)
    write_table(outputs["predictions"], predictions)
    write_json(metrics, outputs["metrics"])
    write_json(summary_payload, outputs["summary"])
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run full structured-sequence baseline artifacts.")
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/baseball_vision")
    parser.add_argument("--clip-run-id", default="mlb_2024_2026_full_v1")
    parser.add_argument("--prediction-run-id", default="sequence_structured_mlb_2024_2026_v1")
    parser.add_argument("--bbe-events", default=None)
    parser.add_argument("--clips", default=None)
    parser.add_argument("--target-registry", default=str(DEFAULT_TARGET_REGISTRY))
    parser.add_argument("--prior-mode", default="past_only")
    parser.add_argument("--aggregation-method", default="quality_weighted_pooling")
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--require-non-empty", action="store_true")
    parser.add_argument("--output-format", choices=("parquet", "jsonl", "json", "csv"), default="parquet")
    parser.add_argument("--structured-sequence-feature-id", default="structured_sequence_v1")
    parser.add_argument("--clip-embedding-feature-id", default="clip_embedding_v1")
    parser.add_argument("--player-season-embedding-feature-id", default="player_season_embedding_v1")
    parser.add_argument("--sequence-dataset-id", default="sequence_dataset_v1")
    parser.add_argument("--event-with-prior-dataset-id", default="event_with_player_prior_v1")
    args = parser.parse_args(argv)
    outputs = run_full_sequence_baseline(
        args.base_dir,
        clip_run_id=args.clip_run_id,
        prediction_run_id=args.prediction_run_id,
        bbe_events=args.bbe_events,
        clips_path=args.clips,
        target_registry=args.target_registry,
        prior_mode=args.prior_mode,
        aggregation_method=args.aggregation_method,
        frame_count=args.frame_count,
        require_non_empty=args.require_non_empty,
        output_suffix="." + args.output_format,
        structured_sequence_feature_id=args.structured_sequence_feature_id,
        clip_embedding_feature_id=args.clip_embedding_feature_id,
        player_season_embedding_feature_id=args.player_season_embedding_feature_id,
        sequence_dataset_id=args.sequence_dataset_id,
        event_with_prior_dataset_id=args.event_with_prior_dataset_id,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
