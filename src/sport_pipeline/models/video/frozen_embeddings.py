"""Colab frozen visual encoder extraction and lightweight-head training.

This module is the heavier counterpart to the local-safe video smoke baseline.
It supports DINO-style contact-frame image embeddings and VideoMAE-style
contact-centered clip embeddings, then trains a small supervised head while
preserving predictions_v1 and optional-target masks.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

from sport_pipeline.artifact_check import write_json
from sport_pipeline.evaluation import evaluate_predictions, load_target_registry, validate_prediction_rows
from sport_pipeline.io import read_table, write_table
from sport_pipeline.io.runtime_cache import cache_file


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TARGET_REGISTRY = PROJECT_ROOT / "configs/targets/target_registry_v1.yaml"
DEFAULT_DINO_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEFAULT_VIDEOMAE_MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"
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


def _resolve_clip_path(clip_row: dict[str, Any], base_dir: Path) -> Path | None:
    raw = clip_row.get("clip_path")
    if _is_missing(raw) or not str(raw):
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = base_dir / path
    return path if path.exists() else None


def _select_representative_clips(clip_rows: list[dict[str, Any]], base_dir: Path, max_clips: int | None) -> list[dict[str, Any]]:
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clip_rows:
        if row.get("clip_status") == "excluded":
            continue
        clip_path = _resolve_clip_path(row, base_dir)
        if clip_path is None:
            continue
        enriched = dict(row)
        enriched["_resolved_clip_path"] = str(clip_path)
        by_event[str(row["event_id"])].append(enriched)
    selected = []
    for rows in by_event.values():
        selected.append(
            sorted(
                rows,
                key=lambda row: (
                    1.0 if row.get("clip_status") == "clean_clip" else 0.0,
                    _to_float(row.get("contact_confidence"), 0.0) + _to_float(row.get("view_confidence"), 0.0),
                    str(row.get("clip_id", "")),
                ),
                reverse=True,
            )[0]
        )
    selected = sorted(selected, key=lambda row: str(row["event_id"]))
    return selected[:max_clips] if max_clips is not None else selected


def _bytes_from_mb(value: int | float | None) -> int | None:
    if value is None:
        return None
    return max(0, int(float(value) * 1024**2))


def _bytes_from_gb(value: int | float | None, default_gb: float = 20.0) -> int:
    if value is None:
        value = default_gb
    return max(0, int(float(value) * 1024**3))


def _cache_selected_clips(
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


def _import_visual_stack():
    try:
        import cv2  # type: ignore
        import torch  # type: ignore
        from PIL import Image  # type: ignore
        from transformers import AutoImageProcessor, AutoModel, AutoModelForVideoClassification  # type: ignore

        return cv2, torch, Image, AutoImageProcessor, AutoModel, AutoModelForVideoClassification
    except ImportError as exc:  # pragma: no cover - Colab dependency path
        raise RuntimeError(
            "Frozen visual encoders require cv2, torch, pillow, and transformers in Colab."
        ) from exc


def _runtime_clip_path(clip_row: dict[str, Any]) -> Path:
    return Path(str(clip_row.get("_runtime_clip_path") or clip_row["_resolved_clip_path"]))


def _require_download_flag(model_id: str, allow_model_download: bool) -> None:
    if Path(model_id).exists():
        return
    if not allow_model_download:
        raise RuntimeError(
            f"Model '{model_id}' may download weights. Re-run in Colab with --allow-model-download."
        )


def _read_uniform_pil_frames(video_path: Path, num_frames: int, image_size: int | None = None) -> list[Any]:
    cv2, _torch, Image, _processor_cls, _model_cls, _video_model_cls = _import_visual_stack()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        total = num_frames
    indices = sorted({int(round(i * max(total - 1, 1) / max(num_frames - 1, 1))) for i in range(num_frames)})
    frames = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        if image_size:
            image = image.resize((image_size, image_size))
        frames.append(image)
    cap.release()
    return frames


def _read_contact_pil_frame(video_path: Path, clip_row: dict[str, Any], image_size: int | None = None) -> Any | None:
    cv2, _torch, Image, _processor_cls, _model_cls, _video_model_cls = _import_visual_stack()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None
    frame_index = int(_to_float(clip_row.get("contact_frame"), 0.0))
    if frame_index <= 0:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_index = max(total // 2, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    if image_size:
        image = image.resize((image_size, image_size))
    return image


def _tensor_to_vector(output: Any) -> list[float]:
    tensor = getattr(output, "pooler_output", None)
    if tensor is None:
        tensor = getattr(output, "last_hidden_state", None)
        if tensor is not None:
            tensor = tensor.mean(dim=1)
    if tensor is None:
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states:
            tensor = hidden_states[-1].mean(dim=1)
    if tensor is None:
        tensor = getattr(output, "logits", None)
    if tensor is None:
        raise RuntimeError("encoder output has no pooler, hidden state, or logits tensor")
    vector = tensor.detach().cpu()[0].float().tolist()
    return [float(value) for value in vector]


def _embedding_encoder_name(encoder: str) -> str:
    if encoder == "dinov3":
        return "dinov3_contact_frame_frozen"
    if encoder == "videomae":
        return "videomae_contact_clip_frozen"
    raise ValueError(f"unknown encoder: {encoder}")


def _load_hf_encoder(
    model_id: str,
    *,
    allow_model_download: bool,
    device: str,
    trust_remote_code: bool,
    prefer_video_model: bool,
) -> tuple[Any, Any]:
    _require_download_flag(model_id, allow_model_download)
    _cv2, _torch, _Image, processor_cls, model_cls, video_model_cls = _import_visual_stack()
    processor = processor_cls.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model_errors: list[str] = []
    # Prefer the base encoder so embeddings come from hidden states rather than
    # classification logits; fall back to video-classification models when needed.
    model_classes = [model_cls, video_model_cls]
    for cls in model_classes:
        try:
            model = cls.from_pretrained(model_id, trust_remote_code=trust_remote_code).to(device).eval()
            return processor, model
        except Exception as exc:  # pragma: no cover - depends on remote model config
            model_errors.append(f"{cls.__name__}: {exc}")
    raise RuntimeError(f"could not load encoder '{model_id}': {' | '.join(model_errors)}")


def extract_dino_embedding(
    clip_row: dict[str, Any],
    *,
    processor: Any,
    model: Any,
    device: str,
    image_size: int | None,
) -> list[float]:
    _cv2, torch, _Image, _processor_cls, _model_cls, _video_model_cls = _import_visual_stack()
    image = _read_contact_pil_frame(_runtime_clip_path(clip_row), clip_row, image_size=image_size)
    if image is None:
        raise RuntimeError("could not read contact frame")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
    return _tensor_to_vector(output)


def extract_videomae_embedding(
    clip_row: dict[str, Any],
    *,
    processor: Any,
    model: Any,
    device: str,
    num_frames: int,
    image_size: int | None,
) -> list[float]:
    _cv2, torch, _Image, _processor_cls, _model_cls, _video_model_cls = _import_visual_stack()
    frames = _read_uniform_pil_frames(_runtime_clip_path(clip_row), num_frames=num_frames, image_size=image_size)
    if not frames:
        raise RuntimeError("could not read clip frames")
    try:
        inputs = processor(frames, return_tensors="pt")
    except TypeError:
        inputs = processor(videos=[frames], return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
    return _tensor_to_vector(output)


def _target_normalizers(
    samples: list[dict[str, Any]],
    target_names: list[str],
    targets: dict[str, Any],
    train_indices: list[int],
) -> dict[str, dict[str, float | str]]:
    normalizers: dict[str, dict[str, float | str]] = {}
    for target_name in target_names:
        target = targets[target_name]
        if target.kind in {"binary", "probability"}:
            normalizers[target_name] = {"kind": target.kind, "mean": 0.0, "std": 1.0}
            continue
        values = [
            float(samples[index].get(target.column))
            for index in train_indices
            if not _is_missing(samples[index].get(target.column))
        ]
        if not values:
            normalizers[target_name] = {"kind": target.kind, "mean": 0.0, "std": 1.0}
            continue
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / max(len(values), 1)
        std_value = max(math.sqrt(variance), 1.0e-6)
        normalizers[target_name] = {"kind": target.kind, "mean": float(mean_value), "std": float(std_value)}
    return normalizers


def _sample_from_embedding(
    clip: dict[str, Any],
    event: dict[str, Any],
    embedding: list[float],
    encoder_name: str,
    encoder_version: str,
    *,
    clip_run_id: str,
    prediction_run_id: str,
) -> dict[str, Any]:
    sample = {
        "schema_version": "visual_frozen_embedding_v1",
        "clip_run_id": clip_run_id,
        "prediction_run_id": prediction_run_id,
        "sample_id": str(clip["clip_id"]),
        "clip_id": clip["clip_id"],
        "event_id": clip["event_id"],
        "same_event_group_id": clip["same_event_group_id"],
        "view_id": clip["view_id"],
        "batter_id": clip["batter_id"],
        "season": clip["season"],
        "batter_season_id": clip["batter_season_id"],
        "clip_path": clip["_resolved_clip_path"],
        "encoder_name": encoder_name,
        "encoder_version": encoder_version,
        "embedding_values": embedding,
        "embedding_dim": len(embedding),
        "clip_status": clip.get("clip_status"),
        "quality_tier": clip.get("quality_tier"),
        "view_label": clip.get("view_label"),
        "view_confidence": _to_float(clip.get("view_confidence"), 0.0),
        "contact_confidence": _to_float(clip.get("contact_confidence"), 0.0),
        "split": clip.get("split", "unknown"),
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
    for target in EVENT_TARGETS:
        sample[f"target_{target}_available"] = bool(event.get(f"target_{target}_available", True))
    return sample


def _train_linear_head(
    samples: list[dict[str, Any]],
    targets: dict[str, Any],
    device: str,
    epochs: int,
    learning_rate: float,
    *,
    progress_callback: Any | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = True,
) -> tuple[dict[str, list[float]], dict[str, dict[str, float | str]], int]:
    if not samples:
        return {}, {}, 0
    _cv2, torch, _Image, _processor_cls, _model_cls, _video_model_cls = _import_visual_stack()
    target_names = [name for name in EVENT_TARGETS if name in targets and targets[name].level == "event"]
    x_tensor = torch.tensor([sample["embedding_values"] for sample in samples], dtype=torch.float32, device=device)
    train_indices = [index for index, sample in enumerate(samples) if sample.get("split") == "train"]
    if not train_indices:
        train_indices = list(range(len(samples)))
    train_index_tensor = torch.tensor(train_indices, dtype=torch.long, device=device)
    normalizers = _target_normalizers(samples, target_names, targets, train_indices)
    y_values = []
    mask_values = []
    for sample in samples:
        y_row = []
        mask_row = []
        for target_name in target_names:
            target = targets[target_name]
            value = sample.get(target.column)
            available = not _is_missing(value)
            if not available:
                y_row.append(0.0)
            elif target.kind in {"binary", "probability"}:
                y_row.append(float(value))
            else:
                normalizer = normalizers[target_name]
                y_row.append((float(value) - float(normalizer["mean"])) / float(normalizer["std"]))
            mask_row.append(1.0 if available else 0.0)
        y_values.append(y_row)
        mask_values.append(mask_row)
    y_tensor = torch.tensor(y_values, dtype=torch.float32, device=device)
    mask_tensor = torch.tensor(mask_values, dtype=torch.float32, device=device)
    model = torch.nn.Sequential(
        torch.nn.LayerNorm(x_tensor.shape[-1]),
        torch.nn.Linear(x_tensor.shape[-1], max(16, min(256, x_tensor.shape[-1] // 2))),
        torch.nn.ReLU(),
        torch.nn.Linear(max(16, min(256, x_tensor.shape[-1] // 2)), len(target_names)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start_epoch = 0
    if resume and checkpoint_path is not None and checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if (
                checkpoint.get("embedding_dim") == int(x_tensor.shape[-1])
                and checkpoint.get("target_names") == target_names
            ):
                model.load_state_dict(checkpoint["model_state"])
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                start_epoch = int(checkpoint.get("epoch", 0))
        except Exception:
            start_epoch = 0

    for epoch in range(start_epoch, max(0, epochs)):
        optimizer.zero_grad()
        logits = model(x_tensor[train_index_tensor])
        losses = []
        for target_index, target_name in enumerate(target_names):
            mask = mask_tensor[train_index_tensor, target_index] > 0
            if not torch.any(mask):
                continue
            if targets[target_name].kind in {"binary", "probability"}:
                losses.append(torch.nn.functional.binary_cross_entropy_with_logits(logits[mask, target_index], y_tensor[train_index_tensor][mask, target_index]))
            elif targets[target_name].loss == "huber":
                losses.append(torch.nn.functional.smooth_l1_loss(logits[mask, target_index], y_tensor[train_index_tensor][mask, target_index]))
            else:
                losses.append(torch.nn.functional.mse_loss(logits[mask, target_index], y_tensor[train_index_tensor][mask, target_index]))
        loss = sum(losses) / len(losses) if losses else logits.sum() * 0.0
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu().item())
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema_version": "frozen_visual_linear_head_checkpoint_v1",
                    "epoch": epoch + 1,
                    "embedding_dim": int(x_tensor.shape[-1]),
                    "target_names": target_names,
                    "normalizers": normalizers,
                    "train_samples": len(train_indices),
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                checkpoint_path,
            )
        if progress_callback is not None:
            progress_callback(epoch + 1, loss_value)
    with torch.no_grad():
        logits = model(x_tensor).detach().cpu()
    outputs: dict[str, list[float]] = {}
    for target_index, target_name in enumerate(target_names):
        values = logits[:, target_index].tolist()
        if targets[target_name].kind in {"binary", "probability"}:
            values = [float(1.0 / (1.0 + math.exp(-value))) for value in values]
        else:
            normalizer = normalizers[target_name]
            values = [float(value) * float(normalizer["std"]) + float(normalizer["mean"]) for value in values]
        outputs[target_name] = values
    return outputs, normalizers, len(train_indices)


def _build_prediction_rows(
    samples: list[dict[str, Any]],
    outputs: dict[str, list[float]],
    targets: dict[str, Any],
    run_id: str,
    aggregation_scope: str,
    model_family: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_name, y_preds in outputs.items():
        target = targets[target_name]
        for sample, y_pred in zip(samples, y_preds):
            value = sample.get(target.column)
            available = not _is_missing(value)
            rows.append(
                {
                    "run_id": run_id,
                    "sample_id": sample["sample_id"],
                    "event_id": sample["event_id"],
                    "batter_season_id": sample["batter_season_id"],
                    "prediction_level": "event",
                    "target_name": target_name,
                    "y_true": None if not available else float(value),
                    "y_pred": None if not available else float(y_pred),
                    "target_available": available,
                    "target_source": target.column,
                    "head_kind": target.kind,
                    "loss_name": target.loss,
                    "aggregation_scope": aggregation_scope,
                    "prior_mode": "none",
                    "label_missing_reason": None if available else sample.get(f"{target_name}_missing_reason", "label_missing"),
                    "requires_pa_manifest": target.requires_pa_manifest,
                    "n_prior_clips": 0,
                    "aggregation_method": model_family,
                    "same_event_ensemble": False,
                    "prediction_std": None,
                    "split": sample.get("split", "unknown"),
                }
            )
    return rows


def run_frozen_visual_encoder(
    base_dir: str | Path,
    *,
    clip_run_id: str = "mlb_2024_2026_full_v1",
    prediction_run_id: str = "video_frozen_encoder_mlb_2024_2026_v1",
    encoder: str = "videomae",
    model_id: str | None = None,
    bbe_events: str | Path | None = None,
    clips_path: str | Path | None = None,
    target_registry: str | Path = DEFAULT_TARGET_REGISTRY,
    allow_model_download: bool = False,
    max_clips: int | None = None,
    num_frames: int = 16,
    image_size: int | None = 224,
    head_epochs: int = 20,
    head_learning_rate: float = 1e-3,
    device: str = "auto",
    trust_remote_code: bool = False,
    require_non_empty: bool = False,
    output_suffix: str = ".parquet",
    resume: bool = True,
    checkpoint_every_clips: int = 1,
    feature_dir_id: str | None = None,
    cache_dir: str | Path | None = None,
    cache_inputs: bool = False,
    cache_num_workers: int = 4,
    cache_min_free_disk_gb: float = 20.0,
    cache_max_file_mb: float | None = None,
) -> dict[str, Path]:
    """Extract frozen visual embeddings and train a small event-level head."""

    base = Path(base_dir)
    targets = load_target_registry(target_registry)
    bbe_path = Path(bbe_events) if bbe_events else base / "manifests/bbe_events_v1.parquet"
    clips_file = Path(clips_path) if clips_path else base / f"clips/{clip_run_id}/clips_v1.parquet"
    events = {str(row["event_id"]): row for row in read_table(bbe_path)}
    clip_rows = read_table(clips_file) if clips_file.exists() else []
    selected = _select_representative_clips(clip_rows, base, max_clips=max_clips)
    selected, cache_stats = _cache_selected_clips(
        selected,
        cache_dir=cache_dir,
        namespace=f"runtime_io/frozen_visual/{prediction_run_id}/clips",
        enabled=cache_inputs,
        num_workers=cache_num_workers,
        max_file_mb=cache_max_file_mb,
        min_free_disk_gb=cache_min_free_disk_gb,
    )
    effective_model = model_id or (DEFAULT_DINO_MODEL if encoder == "dinov3" else DEFAULT_VIDEOMAE_MODEL)
    expected_encoder_name = _embedding_encoder_name(encoder)
    default_feature_dir = "image_embedding_v1" if encoder == "dinov3" else "video_embedding_v1"
    feature_dir = feature_dir_id or default_feature_dir
    outputs = {
        "embeddings": base / f"features/{feature_dir}/manifest{output_suffix}",
        "predictions": base / f"predictions/{prediction_run_id}/predictions_v1{output_suffix}",
        "metrics": base / f"predictions/{prediction_run_id}/metrics_v1.json",
        "summary": base / f"reports/preflight/frozen_visual_encoder_{prediction_run_id}.json",
        "progress": base / f"reports/preflight/frozen_visual_encoder_{prediction_run_id}_progress.json",
        "head_checkpoint": base / f"models/video/{prediction_run_id}/linear_head_checkpoint.pt",
    }
    existing_samples = read_table(outputs["embeddings"]) if resume and outputs["embeddings"].exists() else []
    samples: list[dict[str, Any]] = [
        row
        for row in existing_samples
        if row.get("encoder_name") == expected_encoder_name and row.get("encoder_version") == effective_model
        and str(row.get("clip_run_id") or clip_run_id) == clip_run_id
    ]
    skipped: list[dict[str, Any]] = []
    ignored_existing_embeddings = len(existing_samples) - len(samples)
    if ignored_existing_embeddings:
        skipped.append(
            {
                "reason": "existing_embedding_model_mismatch",
                "ignored_rows": ignored_existing_embeddings,
                "expected_encoder_name": expected_encoder_name,
                "expected_encoder_version": effective_model,
            }
        )
    completed_clip_ids = {str(sample.get("clip_id")) for sample in samples if sample.get("clip_id") is not None}
    pending_selected = [clip for clip in selected if str(clip.get("clip_id")) not in completed_clip_ids]

    def write_progress(status: str, *, seen_clips: int = 0, epoch: int | None = None, loss: float | None = None) -> None:
        payload = {
            "schema_version": "frozen_visual_encoder_progress_v1",
            "status": status,
            "clip_run_id": clip_run_id,
            "prediction_run_id": prediction_run_id,
            "feature_dir_id": feature_dir,
            "encoder": encoder,
            "model_id": effective_model,
            "resume": resume,
            "seen_clips": seen_clips,
            "selected_clips": len(selected),
            "pending_clips": len(pending_selected),
            "completed_clip_ids": sorted(completed_clip_ids),
            "embedding_rows": len(samples),
            "head_epoch": epoch,
            "head_loss": loss,
            "cache_stats": cache_stats,
            "outputs": {key: str(path) for key, path in outputs.items()},
            "skipped_tail": skipped[-100:],
        }
        outputs["progress"].parent.mkdir(parents=True, exist_ok=True)
        outputs["progress"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def checkpoint_embeddings(status: str, seen_clips: int) -> None:
        write_table(outputs["embeddings"], samples)
        write_progress(status, seen_clips=seen_clips)

    selected_device = "no_clip_inputs" if device == "auto" else device
    processor = None
    model = None
    if pending_selected or samples:
        _cv2, torch, _Image, _processor_cls, _model_cls, _video_model_cls = _import_visual_stack()
        selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    if pending_selected:
        processor, model = _load_hf_encoder(
            effective_model,
            allow_model_download=allow_model_download,
            device=selected_device,
            trust_remote_code=trust_remote_code,
            prefer_video_model=(encoder == "videomae"),
        )

    for seen_clips, clip in enumerate(selected, start=1):
        if resume and str(clip.get("clip_id")) in completed_clip_ids:
            if seen_clips % max(1, checkpoint_every_clips) == 0:
                write_progress("running_cached_embeddings", seen_clips=seen_clips)
            continue
        event = events.get(str(clip["event_id"]))
        if event is None:
            skipped.append({"clip_id": clip.get("clip_id"), "reason": "event_not_found"})
            write_progress("running_embeddings", seen_clips=seen_clips)
            continue
        try:
            if encoder == "dinov3":
                embedding = extract_dino_embedding(
                    clip,
                    processor=processor,
                    model=model,
                    device=selected_device,
                    image_size=image_size,
                )
                encoder_name = expected_encoder_name
                aggregation_scope = "image_frozen_encoder"
            elif encoder == "videomae":
                embedding = extract_videomae_embedding(
                    clip,
                    processor=processor,
                    model=model,
                    device=selected_device,
                    num_frames=num_frames,
                    image_size=image_size,
                )
                encoder_name = expected_encoder_name
                aggregation_scope = "video_frozen_encoder"
            else:
                raise ValueError(f"unknown encoder: {encoder}")
        except Exception as exc:
            skipped.append({"clip_id": clip.get("clip_id"), "reason": f"embedding_failed:{exc}"})
            write_progress("running_embeddings", seen_clips=seen_clips)
            continue
        samples.append(
            _sample_from_embedding(
                clip,
                event,
                embedding,
                encoder_name,
                effective_model,
                clip_run_id=clip_run_id,
                prediction_run_id=prediction_run_id,
            )
        )
        completed_clip_ids.add(str(clip["clip_id"]))
        if seen_clips % max(1, checkpoint_every_clips) == 0:
            checkpoint_embeddings("running_embeddings", seen_clips)

    if samples:
        checkpoint_embeddings("embeddings_complete", len(selected))

    outputs_by_target, target_normalizers, train_samples = _train_linear_head(
        samples,
        targets,
        device=selected_device,
        epochs=head_epochs,
        learning_rate=head_learning_rate,
        progress_callback=lambda epoch, loss: write_progress("training_head", seen_clips=len(selected), epoch=epoch, loss=loss),
        checkpoint_path=outputs["head_checkpoint"],
        resume=resume,
    )
    aggregation_scope = "image_frozen_encoder" if encoder == "dinov3" else "video_frozen_encoder"
    predictions = _build_prediction_rows(
        samples,
        outputs_by_target,
        targets,
        run_id=prediction_run_id,
        aggregation_scope=aggregation_scope,
        model_family=f"{encoder}_frozen_linear_head",
    )
    validate_prediction_rows(predictions)
    metrics = evaluate_predictions(predictions, targets, run_id=prediction_run_id)
    summary_payload = {
        "schema_version": "frozen_visual_encoder_summary_v1",
        "clip_run_id": clip_run_id,
        "prediction_run_id": prediction_run_id,
        "feature_dir_id": feature_dir,
        "encoder": encoder,
        "model_id": effective_model,
        "device": selected_device,
        "allow_model_download": allow_model_download,
        "trust_remote_code": trust_remote_code,
        "require_non_empty": require_non_empty,
        "resume": resume,
        "checkpoint_every_clips": checkpoint_every_clips,
        "cache_dir": None if cache_dir is None else str(cache_dir),
        "cache_inputs": cache_inputs,
        "cache_num_workers": cache_num_workers,
        "cache_min_free_disk_gb": cache_min_free_disk_gb,
        "cache_max_file_mb": cache_max_file_mb,
        "cache_stats": cache_stats,
        "progress_path": str(outputs["progress"]),
        "head_checkpoint_path": str(outputs["head_checkpoint"]),
        "input_clips": len(clip_rows),
        "selected_clips": len(selected),
        "embedding_rows": len(samples),
        "prediction_rows": len(predictions),
        "train_samples": train_samples,
        "target_normalizers": target_normalizers,
        "head_epochs": head_epochs,
        "skipped": skipped[:100],
        "outputs": {key: str(path) for key, path in outputs.items() if key != "summary"},
    }
    if require_non_empty and not samples:
        write_json(summary_payload, outputs["summary"])
        write_progress("failed_empty", seen_clips=len(selected))
        raise RuntimeError(
            "frozen visual encoder produced 0 embedding samples; not writing empty visual artifacts in real-run mode. "
            "Check clips_v1 has real clip_path files, model dependencies are installed, and ALLOW_MODEL_DOWNLOAD is enabled if needed. "
            f"summary_path={outputs['summary']}"
        )
    write_table(outputs["embeddings"], samples)
    write_table(outputs["predictions"], predictions)
    write_json(metrics, outputs["metrics"])
    write_json(summary_payload, outputs["summary"])
    write_progress("complete", seen_clips=len(selected))
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run DINO/VideoMAE frozen visual baseline in Colab.")
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/baseball_vision")
    parser.add_argument("--clip-run-id", default="mlb_2024_2026_full_v1")
    parser.add_argument("--prediction-run-id", default="video_frozen_encoder_mlb_2024_2026_v1")
    parser.add_argument("--encoder", choices=("videomae", "dinov3"), default="videomae")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--bbe-events", default=None)
    parser.add_argument("--clips", default=None)
    parser.add_argument("--target-registry", default=str(DEFAULT_TARGET_REGISTRY))
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--head-epochs", type=int, default=20)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--require-non-empty", action="store_true")
    parser.add_argument("--output-format", choices=("parquet", "jsonl", "json", "csv"), default="parquet")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--checkpoint-every-clips", type=int, default=1)
    parser.add_argument("--feature-dir-id", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--cache-inputs", action="store_true")
    parser.add_argument("--cache-num-workers", type=int, default=4)
    parser.add_argument("--cache-min-free-disk-gb", type=float, default=20.0)
    parser.add_argument("--cache-max-file-mb", type=float, default=None)
    args = parser.parse_args(argv)
    outputs = run_frozen_visual_encoder(
        args.base_dir,
        clip_run_id=args.clip_run_id,
        prediction_run_id=args.prediction_run_id,
        encoder=args.encoder,
        model_id=args.model_id,
        bbe_events=args.bbe_events,
        clips_path=args.clips,
        target_registry=args.target_registry,
        allow_model_download=args.allow_model_download,
        max_clips=args.max_clips,
        num_frames=args.num_frames,
        image_size=args.image_size,
        head_epochs=args.head_epochs,
        head_learning_rate=args.head_learning_rate,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        require_non_empty=args.require_non_empty,
        output_suffix="." + args.output_format,
        resume=not args.no_resume,
        checkpoint_every_clips=args.checkpoint_every_clips,
        feature_dir_id=args.feature_dir_id,
        cache_dir=args.cache_dir,
        cache_inputs=args.cache_inputs,
        cache_num_workers=args.cache_num_workers,
        cache_min_free_disk_gb=args.cache_min_free_disk_gb,
        cache_max_file_mb=args.cache_max_file_mb,
    )
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
