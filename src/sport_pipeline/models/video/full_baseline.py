"""Full video/image baseline runner for Colab artifacts.

The default encoder mode uses lightweight OpenCV-derived clip features so the
pipeline can be tested end-to-end without model downloads. A future Colab-only
VideoMAE encoder can replace the feature extractor while preserving the same
predictions_v1 output contract.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from sport_pipeline.artifact_check import write_json
from sport_pipeline.evaluation import evaluate_predictions, validate_prediction_rows
from sport_pipeline.io import read_table, write_table
from sport_pipeline.io.runtime_cache import cache_file
from sport_pipeline.models.video.interface import FrozenVideoBaseline
from sport_pipeline.models.video.predictions import build_visual_prediction_rows


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TARGET_REGISTRY = PROJECT_ROOT / "configs/targets/target_registry_v1.yaml"


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


def _resolve_clip_path(clip_row: dict[str, Any], base_dir: Path) -> Path | None:
    raw = clip_row.get("clip_path")
    if _is_missing(raw) or not str(raw):
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = base_dir / path
    return path if path.exists() else None


def _read_frames_cv2(video_path: Path, max_frames: int) -> list[Any]:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for lightweight video feature extraction in Colab") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        total = max_frames
    indices = sorted({int(round(i * max(total - 1, 1) / max(max_frames - 1, 1))) for i in range(max_frames)})
    frames: list[Any] = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(frame)
    cap.release()
    return frames


def extract_lightweight_video_features(video_path: Path, max_frames: int = 16) -> list[float]:
    """Extract deterministic frame statistics from a clip."""

    frames = _read_frames_cv2(video_path, max_frames=max_frames)
    if not frames:
        raise RuntimeError(f"no frames read from clip: {video_path}")
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    resized = [cv2.resize(frame, (64, 64)) for frame in frames]
    arr = np.stack(resized).astype("float32") / 255.0
    rgb = arr[:, :, :, ::-1]
    channel_mean = rgb.mean(axis=(0, 1, 2)).tolist()
    channel_std = rgb.std(axis=(0, 1, 2)).tolist()
    gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0 for frame in resized]
    motion = [float(np.mean(np.abs(gray[i] - gray[i - 1]))) for i in range(1, len(gray))]
    motion_mean = mean(motion) if motion else 0.0
    motion_max = max(motion) if motion else 0.0
    return [float(value) for value in channel_mean + channel_std + [motion_mean, motion_max]]


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


def _select_representative_video_clips(clip_rows: list[dict[str, Any]], base_dir: Path, max_clips: int | None) -> list[dict[str, Any]]:
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clip_rows:
        if row.get("clip_status") == "excluded":
            continue
        clip_path = _resolve_clip_path(row, base_dir)
        if clip_path is None:
            continue
        output = dict(row)
        output["_resolved_clip_path"] = str(clip_path)
        by_event[str(row["event_id"])].append(output)
    selected = [sorted(rows, key=_clip_score, reverse=True)[0] for rows in by_event.values()]
    selected = sorted(selected, key=lambda row: str(row.get("event_id", "")))
    if max_clips is not None:
        selected = selected[:max_clips]
    return selected


def _bytes_from_mb(value: int | float | None) -> int | None:
    if value is None:
        return None
    return max(0, int(float(value) * 1024**2))


def _bytes_from_gb(value: int | float | None, default_gb: float = 20.0) -> int:
    if value is None:
        value = default_gb
    return max(0, int(float(value) * 1024**3))


def _cache_selected_video_clips(
    selected: list[dict[str, Any]],
    *,
    cache_dir: str | Path | None,
    namespace: str,
    enabled: bool,
    num_workers: int,
    max_file_mb: float | None,
    min_free_disk_gb: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stats: dict[str, Any] = {"enabled": bool(enabled and cache_dir is not None), "used": 0, "reasons": {}}
    if not enabled or cache_dir is None or not selected:
        return selected, stats
    max_file_bytes = _bytes_from_mb(max_file_mb)
    min_free_bytes = _bytes_from_gb(min_free_disk_gb)

    def stage(index: int, clip: dict[str, Any]) -> tuple[int, dict[str, Any], str, bool]:
        result = cache_file(
            clip["_resolved_clip_path"],
            cache_dir=cache_dir,
            namespace=namespace,
            key=str(clip.get("clip_id") or index),
            enabled=True,
            max_file_bytes=max_file_bytes,
            min_free_disk_bytes=min_free_bytes,
        )
        staged = dict(clip)
        staged["_runtime_clip_path"] = str(result.path)
        return index, staged, result.reason, result.used_cache

    max_workers = max(1, int(num_workers or 1))
    if max_workers == 1:
        results = [stage(index, clip) for index, clip in enumerate(selected)]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(stage, index, clip) for index, clip in enumerate(selected)]
            for future in as_completed(futures):
                results.append(future.result())
    staged_by_index: dict[int, dict[str, Any]] = {}
    for index, staged, reason, used in results:
        staged_by_index[index] = staged
        stats["reasons"][reason] = int(stats["reasons"].get(reason, 0)) + 1
        if used:
            stats["used"] += 1
    return [staged_by_index.get(index, clip) for index, clip in enumerate(selected)], stats


def _build_video_sample(row: dict[str, Any], event: dict[str, Any], feature_values: list[float]) -> dict[str, Any]:
    sample = {
        "schema_version": "video_embedding_v1",
        "sample_id": str(row["clip_id"]),
        "clip_id": row["clip_id"],
        "event_id": row["event_id"],
        "same_event_group_id": row["same_event_group_id"],
        "view_id": row["view_id"],
        "batter_id": row["batter_id"],
        "season": row["season"],
        "batter_season_id": row["batter_season_id"],
        "clip_path": row.get("_resolved_clip_path"),
        "encoder_name": "lightweight_cv2_video_features",
        "encoder_version": "full_baseline_v1",
        "embedding_values": feature_values,
        "embedding_dim": len(feature_values),
        "clip_status": row.get("clip_status"),
        "quality_tier": row.get("quality_tier"),
        "view_label": row.get("view_label"),
        "view_confidence": _to_float(row.get("view_confidence"), 0.0),
        "contact_confidence": _to_float(row.get("contact_confidence"), 0.0),
        "launch_speed": event.get("launch_speed"),
        "launch_angle": event.get("launch_angle"),
        "target_hard_hit": event.get("target_hard_hit"),
        "target_barrel": event.get("target_barrel"),
        "estimated_ba_using_speedangle": event.get("estimated_ba_using_speedangle"),
        "estimated_woba_using_speedangle": event.get("estimated_woba_using_speedangle"),
        "target_ev_available": bool(event.get("target_ev_available", event.get("launch_speed") is not None)),
        "target_la_available": bool(event.get("target_la_available", event.get("launch_angle") is not None)),
        "target_hard_hit_available": bool(event.get("target_hard_hit_available", event.get("target_hard_hit") is not None)),
        "target_barrel_available": bool(event.get("target_barrel_available", event.get("target_barrel") is not None)),
        "target_xba_available": bool(event.get("target_xba_available", event.get("estimated_ba_using_speedangle") is not None)),
        "target_xwoba_available": bool(event.get("target_xwoba_available", event.get("estimated_woba_using_speedangle") is not None)),
    }
    if not sample["target_xba_available"]:
        sample["xba_missing_reason"] = event.get("label_missing_reason") or "statcast_expected_outcome_missing"
    if not sample["target_xwoba_available"]:
        sample["xwoba_missing_reason"] = event.get("label_missing_reason") or "statcast_expected_outcome_missing"
    return sample


def run_full_video_baseline(
    base_dir: str | Path,
    *,
    clip_run_id: str = "mlb_2024_2026_full_v1",
    prediction_run_id: str = "video_frozen_encoder_mlb_2024_2026_v1",
    bbe_events: str | Path | None = None,
    clips_path: str | Path | None = None,
    target_registry: str | Path = DEFAULT_TARGET_REGISTRY,
    max_clips: int | None = None,
    max_frames: int = 16,
    encoder_mode: str = "lightweight",
    allow_model_download: bool = False,
    require_non_empty: bool = False,
    output_suffix: str = ".parquet",
    video_embedding_feature_id: str | None = None,
    cache_dir: str | Path | None = None,
    cache_inputs: bool = False,
    cache_num_workers: int = 4,
    cache_min_free_disk_gb: float = 20.0,
    cache_max_file_mb: float | None = None,
) -> dict[str, Path]:
    """Run a video baseline from contact-aligned clips into predictions_v1."""

    resolved_feature_id = video_embedding_feature_id or (
        "video_lightweight_features_v1" if encoder_mode == "lightweight" else "video_embedding_v1"
    )
    if encoder_mode != "lightweight":
        from sport_pipeline.models.video.frozen_embeddings import run_frozen_visual_encoder

        return run_frozen_visual_encoder(
            base_dir,
            clip_run_id=clip_run_id,
            prediction_run_id=prediction_run_id,
            encoder="dinov3" if encoder_mode == "dinov3" else "videomae",
            bbe_events=bbe_events,
            clips_path=clips_path,
            target_registry=target_registry,
            allow_model_download=allow_model_download,
            max_clips=max_clips,
            num_frames=max_frames,
            device="auto",
            require_non_empty=require_non_empty,
            output_suffix=output_suffix,
            feature_dir_id=resolved_feature_id,
            cache_dir=cache_dir,
            cache_inputs=cache_inputs,
            cache_num_workers=cache_num_workers,
            cache_min_free_disk_gb=cache_min_free_disk_gb,
            cache_max_file_mb=cache_max_file_mb,
        )

    base = Path(base_dir)
    bbe_path = Path(bbe_events) if bbe_events else base / "manifests/bbe_events_v1.parquet"
    clips = Path(clips_path) if clips_path else base / f"clips/{clip_run_id}/clips_v1.parquet"
    bbe_rows = read_table(bbe_path)
    clip_rows = read_table(clips) if clips.exists() else []
    events = {str(row["event_id"]): row for row in bbe_rows}
    selected = _select_representative_video_clips(clip_rows, base, max_clips=max_clips)
    selected, cache_stats = _cache_selected_video_clips(
        selected,
        cache_dir=cache_dir,
        namespace=f"runtime_io/lightweight_video/{prediction_run_id}/clips",
        enabled=cache_inputs,
        num_workers=cache_num_workers,
        max_file_mb=cache_max_file_mb,
        min_free_disk_gb=cache_min_free_disk_gb,
    )

    samples: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for clip in selected:
        event = events.get(str(clip["event_id"]))
        if event is None:
            skipped.append({"clip_id": clip.get("clip_id"), "reason": "event_not_found"})
            continue
        try:
            features = extract_lightweight_video_features(
                Path(str(clip.get("_runtime_clip_path") or clip["_resolved_clip_path"])),
                max_frames=max_frames,
            )
        except Exception as exc:
            skipped.append({"clip_id": clip.get("clip_id"), "reason": f"feature_extraction_failed:{exc}"})
            continue
        samples.append(_build_video_sample(clip, event, features))

    feature_rows = [
        {
            key: value
            for key, value in sample.items()
            if key
            in {
                "schema_version",
                "sample_id",
                "clip_id",
                "event_id",
                "same_event_group_id",
                "view_id",
                "batter_id",
                "season",
                "batter_season_id",
                "clip_path",
                "encoder_name",
                "encoder_version",
                "embedding_values",
                "embedding_dim",
                "clip_status",
                "quality_tier",
                "view_label",
                "view_confidence",
                "contact_confidence",
            }
        }
        for sample in samples
    ]

    predictions: list[dict[str, Any]] = []
    model = FrozenVideoBaseline(input_dim=8, target_registry_path=target_registry)
    if samples:
        features = [sample["embedding_values"] for sample in samples]
        outputs = model.predict_from_features(features)
        predictions = build_visual_prediction_rows(
            run_id=prediction_run_id,
            samples=samples,
            predictions=outputs,
            head_specs=model.head_specs,
            model_family="lightweight_cv2_video_features",
            aggregation_scope="raw_video_lightweight",
            loss_masks=model.loss_masks(samples),
        )
    validate_prediction_rows(predictions)
    metrics = evaluate_predictions(predictions, model.targets, run_id=prediction_run_id)

    outputs = {
        "video_embeddings": base / f"features/{resolved_feature_id}/manifest{output_suffix}",
        "predictions": base / f"predictions/{prediction_run_id}/predictions_v1{output_suffix}",
        "metrics": base / f"predictions/{prediction_run_id}/metrics_v1.json",
        "summary": base / f"reports/preflight/full_video_baseline_{prediction_run_id}.json",
    }
    summary_payload = {
        "schema_version": "full_video_baseline_summary_v1",
        "clip_run_id": clip_run_id,
        "prediction_run_id": prediction_run_id,
        "encoder_mode": encoder_mode,
        "video_embedding_feature_id": resolved_feature_id,
        "allow_model_download": allow_model_download,
        "require_non_empty": require_non_empty,
        "input_clips": len(clip_rows),
        "selected_video_clips": len(selected),
        "sample_rows": len(samples),
        "prediction_rows": len(predictions),
        "cache_dir": None if cache_dir is None else str(cache_dir),
        "cache_inputs": cache_inputs,
        "cache_num_workers": cache_num_workers,
        "cache_min_free_disk_gb": cache_min_free_disk_gb,
        "cache_max_file_mb": cache_max_file_mb,
        "cache_stats": cache_stats,
        "skipped": skipped[:100],
    }
    if require_non_empty and not samples:
        write_json(summary_payload, outputs["summary"])
        raise RuntimeError(
            "video baseline produced 0 samples; not writing empty video/prediction artifacts in real-run mode. "
            "Check clips_v1 has real clip_path files and rerun 12 if needed. "
            f"summary_path={outputs['summary']}"
        )
    write_table(outputs["video_embeddings"], feature_rows)
    write_table(outputs["predictions"], predictions)
    write_json(metrics, outputs["metrics"])
    write_json(summary_payload, outputs["summary"])
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run full video baseline artifacts.")
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/baseball_vision")
    parser.add_argument("--clip-run-id", default="mlb_2024_2026_full_v1")
    parser.add_argument("--prediction-run-id", default="video_frozen_encoder_mlb_2024_2026_v1")
    parser.add_argument("--bbe-events", default=None)
    parser.add_argument("--clips", default=None)
    parser.add_argument("--target-registry", default=str(DEFAULT_TARGET_REGISTRY))
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--encoder-mode", choices=("lightweight", "videomae", "dinov3"), default="lightweight")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--require-non-empty", action="store_true")
    parser.add_argument("--output-format", choices=("parquet", "jsonl", "json", "csv"), default="parquet")
    parser.add_argument("--video-embedding-feature-id", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--cache-inputs", action="store_true")
    parser.add_argument("--cache-num-workers", type=int, default=4)
    parser.add_argument("--cache-min-free-disk-gb", type=float, default=20.0)
    parser.add_argument("--cache-max-file-mb", type=float, default=None)
    args = parser.parse_args(argv)
    outputs = run_full_video_baseline(
        args.base_dir,
        clip_run_id=args.clip_run_id,
        prediction_run_id=args.prediction_run_id,
        bbe_events=args.bbe_events,
        clips_path=args.clips,
        target_registry=args.target_registry,
        max_clips=args.max_clips,
        max_frames=args.max_frames,
        encoder_mode=args.encoder_mode,
        allow_model_download=args.allow_model_download,
        require_non_empty=args.require_non_empty,
        output_suffix="." + args.output_format,
        video_embedding_feature_id=args.video_embedding_feature_id,
        cache_dir=args.cache_dir,
        cache_inputs=args.cache_inputs,
        cache_num_workers=args.cache_num_workers,
        cache_min_free_disk_gb=args.cache_min_free_disk_gb,
        cache_max_file_mb=args.cache_max_file_mb,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
