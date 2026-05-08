"""Tiny raw-video fine-tuning baseline for Colab ablations.

This module intentionally keeps the model small. It answers a specific
research question: does a trainable model that sees the contact-aligned video
frames beat frozen/raw-feature baselines and structured pose/CV features?
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
EVENT_TARGETS = ("ev", "la", "hard_hit", "barrel", "xba", "xwoba")
KINETICS_RGB_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_RGB_STD = (0.22803, 0.22145, 0.21699)


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


def _resolve_clip_path(clip_row: dict[str, Any], base_dir: Path) -> Path | None:
    raw = clip_row.get("clip_path")
    if _is_missing(raw) or not str(raw):
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = base_dir / path
    return path if path.exists() else None


def _clip_score(row: dict[str, Any]) -> tuple[float, float, float, str]:
    return (
        1.0 if row.get("clip_status") == "clean_clip" else 0.0,
        1.0 if row.get("quality_tier") == "usable_primary" else 0.0,
        _to_float(row.get("contact_confidence"), 0.0) + _to_float(row.get("view_confidence"), 0.0),
        str(row.get("clip_id", "")),
    )


def _select_representative_clips(clip_rows: list[dict[str, Any]], base_dir: Path, max_clips: int | None) -> list[dict[str, Any]]:
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
    return selected[:max_clips] if max_clips is not None else selected


def _event_value(event: dict[str, Any], target: Any) -> Any:
    return event.get(target.column)


def _target_missing_reason(event: dict[str, Any], target_name: str) -> str:
    return str(
        event.get(f"target_{target_name}_missing_reason")
        or event.get("label_missing_reason")
        or "label_missing"
    )


def _build_samples(
    *,
    base_dir: Path,
    clip_run_id: str,
    max_clips: int | None,
    bbe_events: str | Path | None,
    clips_path: str | Path | None,
    target_registry: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    bbe_file = Path(bbe_events) if bbe_events else base_dir / "manifests/bbe_events_v1.parquet"
    clips_file = Path(clips_path) if clips_path else base_dir / f"clips/{clip_run_id}/clips_v1.parquet"
    targets = load_target_registry(target_registry)
    target_names = [name for name in EVENT_TARGETS if name in targets and targets[name].level == "event"]
    events = {str(row["event_id"]): row for row in read_table(bbe_file)}
    clip_rows = read_table(clips_file) if clips_file.exists() else []
    splits = _split_map(base_dir)
    selected = _select_representative_clips(clip_rows, base_dir, max_clips=max_clips)
    samples: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for clip in selected:
        event = events.get(str(clip["event_id"]))
        if event is None:
            skipped.append({"clip_id": clip.get("clip_id"), "reason": "event_not_found"})
            continue
        y_values = []
        masks = []
        missing_reasons = {}
        for target_name in target_names:
            target = targets[target_name]
            value = _event_value(event, target)
            if _is_missing(value):
                y_values.append(0.0)
                masks.append(0.0)
                missing_reasons[target_name] = _target_missing_reason(event, target_name)
            else:
                y_values.append(float(value))
                masks.append(1.0)
        samples.append(
            {
                "sample_id": str(clip["clip_id"]),
                "clip_id": str(clip["clip_id"]),
                "event_id": str(clip["event_id"]),
                "batter_season_id": str(clip["batter_season_id"]),
                "clip_path": str(clip["_resolved_clip_path"]),
                "split": splits.get(str(clip["event_id"]), str(clip.get("split") or "unknown")),
                "event": event,
                "target_names": target_names,
                "y_values": y_values,
                "masks": masks,
                "missing_reasons": missing_reasons,
            }
        )
    input_summary = {
        "bbe_events_path": str(bbe_file),
        "clips_path": str(clips_file),
        "input_clips": len(clip_rows),
        "selected_clips": len(selected),
        "samples": len(samples),
        "skipped": skipped[:100],
        "target_names": target_names,
    }
    return samples, targets, input_summary


def _bytes_from_mb(value: int | float | None) -> int | None:
    if value is None:
        return None
    return max(0, int(float(value) * 1024**2))


def _bytes_from_gb(value: int | float | None, default_gb: float = 20.0) -> int:
    if value is None:
        value = default_gb
    return max(0, int(float(value) * 1024**3))


def _cache_samples_for_runtime(
    samples: list[dict[str, Any]],
    *,
    cache_dir: str | Path | None,
    namespace: str,
    enabled: bool,
    num_workers: int,
    max_file_mb: float | None,
    min_free_disk_gb: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stats: dict[str, Any] = {"enabled": bool(enabled and cache_dir is not None), "used": 0, "reasons": {}}
    if not enabled or cache_dir is None or not samples:
        return samples, stats
    max_file_bytes = _bytes_from_mb(max_file_mb)
    min_free_bytes = _bytes_from_gb(min_free_disk_gb)

    def stage(index: int, sample: dict[str, Any]) -> tuple[int, dict[str, Any], str, bool]:
        result = cache_file(
            sample["clip_path"],
            cache_dir=cache_dir,
            namespace=namespace,
            key=str(sample.get("clip_id") or sample.get("sample_id") or index),
            enabled=True,
            max_file_bytes=max_file_bytes,
            min_free_disk_bytes=min_free_bytes,
        )
        staged = dict(sample)
        staged["runtime_clip_path"] = str(result.path)
        return index, staged, result.reason, result.used_cache

    max_workers = max(1, int(num_workers or 1))
    if max_workers == 1:
        results = [stage(index, sample) for index, sample in enumerate(samples)]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(stage, index, sample) for index, sample in enumerate(samples)]
            for future in as_completed(futures):
                results.append(future.result())
    staged_by_index: dict[int, dict[str, Any]] = {}
    for index, staged, reason, used in results:
        staged_by_index[index] = staged
        stats["reasons"][reason] = int(stats["reasons"].get(reason, 0)) + 1
        if used:
            stats["used"] += 1
    return [staged_by_index.get(index, sample) for index, sample in enumerate(samples)], stats


def _read_video_array(video_path: Path, *, num_frames: int, image_size: int) -> Any:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - Colab dependency path
        raise RuntimeError("OpenCV and NumPy are required for raw video fine-tuning.") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"could not open video: {video_path}")
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
        frame = cv2.resize(frame, (image_size, image_size))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from video: {video_path}")
    while len(frames) < num_frames:
        frames.append(frames[-1])
    arr = np.stack(frames[:num_frames]).astype("float32") / 255.0
    mean = np.array(KINETICS_RGB_MEAN, dtype="float32").reshape(1, 1, 1, 3)
    std = np.array(KINETICS_RGB_STD, dtype="float32").reshape(1, 1, 1, 3)
    arr = (arr - mean) / std
    return arr.transpose(0, 3, 1, 2)


def _import_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        import torch.nn.functional as functional  # type: ignore
        from torch.utils.data import DataLoader, Dataset  # type: ignore

        return torch, nn, functional, DataLoader, Dataset
    except ImportError as exc:  # pragma: no cover - Colab dependency path
        raise RuntimeError("PyTorch is required for raw video fine-tuning. Use a Colab GPU runtime.") from exc


def _make_dataset_class(num_frames: int, image_size: int):
    torch, _nn, _functional, _DataLoader, Dataset = _import_torch()

    class RawVideoDataset(Dataset):  # type: ignore[misc, valid-type]
        def __init__(self, samples: list[dict[str, Any]]) -> None:
            self.samples = samples

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int) -> dict[str, Any]:
            sample = self.samples[index]
            frames = _read_video_array(
                Path(str(sample.get("runtime_clip_path") or sample["clip_path"])),
                num_frames=num_frames,
                image_size=image_size,
            )
            return {
                "index": index,
                "video": torch.tensor(frames, dtype=torch.float32),
                "y": torch.tensor(sample["y_values"], dtype=torch.float32),
                "mask": torch.tensor(sample["masks"], dtype=torch.float32),
            }

    return RawVideoDataset


def _build_tiny3d_model(output_dim: int, hidden_dim: int, dropout: float):
    _torch, nn, _functional, _DataLoader, _Dataset = _import_torch()

    class TinyRawVideoStatNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv3d(3, 16, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
                nn.BatchNorm3d(16),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(1, 2, 2)),
                nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(2, 2, 2)),
                nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, video):  # type: ignore[no-untyped-def]
            # Dataset gives N,T,C,H,W. Conv3d expects N,C,T,H,W.
            return self.head(self.encoder(video.transpose(1, 2)))

    return TinyRawVideoStatNet()


def _build_torchvision_r3d18_model(
    *,
    output_dim: int,
    dropout: float,
    pretrained: bool,
    freeze_backbone: bool,
    allow_model_download: bool,
):
    _torch, nn, _functional, _DataLoader, _Dataset = _import_torch()
    if pretrained and not allow_model_download:
        raise RuntimeError("torchvision r3d_18 pretrained weights may download. Set allow_model_download=True in Colab.")
    try:
        from torchvision.models.video import r3d_18  # type: ignore
    except ImportError as exc:  # pragma: no cover - Colab dependency path
        raise RuntimeError("torchvision is required for model_family='torchvision_r3d18'.") from exc

    weights = None
    if pretrained:
        try:
            from torchvision.models.video import R3D_18_Weights  # type: ignore

            weights = R3D_18_Weights.DEFAULT
        except Exception:  # pragma: no cover - depends on torchvision version
            weights = "DEFAULT"
    try:
        backbone = r3d_18(weights=weights)
    except (TypeError, ValueError):  # pragma: no cover - older torchvision
        backbone = r3d_18(pretrained=pretrained)
    in_features = int(backbone.fc.in_features)
    backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, output_dim))
    if freeze_backbone:
        for name, parameter in backbone.named_parameters():
            parameter.requires_grad = name.startswith("fc.")

    class TorchvisionVideoWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone

        def forward(self, video):  # type: ignore[no-untyped-def]
            # Dataset gives N,T,C,H,W. Torchvision video models expect N,C,T,H,W.
            return self.backbone(video.transpose(1, 2))

    return TorchvisionVideoWrapper()


def _build_model(
    *,
    model_family: str,
    output_dim: int,
    hidden_dim: int,
    dropout: float,
    pretrained: bool,
    freeze_backbone: bool,
    allow_model_download: bool,
):
    if model_family == "tiny3d":
        return _build_tiny3d_model(output_dim=output_dim, hidden_dim=hidden_dim, dropout=dropout)
    if model_family == "torchvision_r3d18":
        return _build_torchvision_r3d18_model(
            output_dim=output_dim,
            dropout=dropout,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            allow_model_download=allow_model_download,
        )
    raise ValueError(f"unknown raw video model_family: {model_family}")


def _train_indices(samples: list[dict[str, Any]]) -> list[int]:
    train = [index for index, sample in enumerate(samples) if sample.get("split") == "train"]
    return train or list(range(len(samples)))


def _target_normalizers(samples: list[dict[str, Any]], target_names: list[str], targets: dict[str, Any], train_indices: list[int]) -> dict[str, dict[str, float | str]]:
    normalizers: dict[str, dict[str, float | str]] = {}
    for target_index, target_name in enumerate(target_names):
        target = targets[target_name]
        if target.kind in {"binary", "probability"}:
            normalizers[target_name] = {"kind": target.kind, "mean": 0.0, "std": 1.0}
            continue
        values = [
            float(samples[index]["y_values"][target_index])
            for index in train_indices
            if float(samples[index]["masks"][target_index]) > 0.0
        ]
        if not values:
            normalizers[target_name] = {"kind": target.kind, "mean": 0.0, "std": 1.0}
            continue
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        normalizers[target_name] = {"kind": target.kind, "mean": mean_value, "std": max(variance ** 0.5, 1.0e-6)}
    return normalizers


def _normalize_targets(y_tensor: Any, target_names: list[str], targets: dict[str, Any], normalizers: dict[str, dict[str, float | str]]) -> Any:
    normalized = y_tensor.clone()
    for index, target_name in enumerate(target_names):
        if targets[target_name].kind in {"binary", "probability"}:
            continue
        normalizer = normalizers[target_name]
        normalized[:, index] = (normalized[:, index] - float(normalizer["mean"])) / float(normalizer["std"])
    return normalized


def _loss(logits: Any, y_tensor: Any, mask_tensor: Any, target_names: list[str], targets: dict[str, Any]) -> Any:
    _torch, _nn, functional, _DataLoader, _Dataset = _import_torch()
    losses = []
    for index, target_name in enumerate(target_names):
        mask = mask_tensor[:, index] > 0
        if not bool(mask.any()):
            continue
        target = targets[target_name]
        pred = logits[mask, index]
        truth = y_tensor[mask, index]
        if target.kind == "binary":
            losses.append(functional.binary_cross_entropy_with_logits(pred, truth))
        elif target.kind == "probability":
            losses.append(functional.mse_loss(pred.sigmoid(), truth.clamp(0.0, 1.0)))
        elif target.loss == "huber":
            losses.append(functional.smooth_l1_loss(pred, truth))
        else:
            losses.append(functional.mse_loss(pred, truth))
    return sum(losses) / len(losses) if losses else logits.sum() * 0.0


def _inverse_prediction(raw_value: float, target_name: str, targets: dict[str, Any], normalizers: dict[str, dict[str, float | str]]) -> float:
    target = targets[target_name]
    if target.kind in {"binary", "probability"}:
        return 1.0 / (1.0 + math.exp(-raw_value))
    normalizer = normalizers[target_name]
    return raw_value * float(normalizer["std"]) + float(normalizer["mean"])


def _prediction_rows(
    *,
    samples: list[dict[str, Any]],
    logits_by_index: dict[int, list[float]],
    target_names: list[str],
    targets: dict[str, Any],
    normalizers: dict[str, dict[str, float | str]],
    prediction_run_id: str,
    model_family: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        logits = logits_by_index.get(sample_index)
        for target_index, target_name in enumerate(target_names):
            target = targets[target_name]
            value = sample["event"].get(target.column)
            label_available = not _is_missing(value)
            trained_for_target = any(float(row["masks"][target_index]) > 0.0 for row in samples)
            available = label_available and trained_for_target and logits is not None
            y_pred = None
            reason = None
            if available:
                y_pred = _inverse_prediction(float(logits[target_index]), target_name, targets, normalizers)
            elif not label_available:
                reason = sample["missing_reasons"].get(target_name) or _target_missing_reason(sample["event"], target_name)
            elif logits is None:
                reason = "raw_video_decode_or_prediction_missing"
            else:
                reason = "raw_video_finetune_not_fit_for_target"
            rows.append(
                {
                    "run_id": prediction_run_id,
                    "sample_id": sample["sample_id"],
                    "event_id": sample["event_id"],
                    "batter_season_id": sample["batter_season_id"],
                    "prediction_level": "event",
                    "target_name": target_name,
                    "y_true": None if not label_available else float(value),
                    "y_pred": y_pred,
                    "target_available": available,
                    "target_source": target.column,
                    "head_kind": target.kind,
                    "loss_name": target.loss,
                    "aggregation_scope": "raw_video_finetune",
                    "prior_mode": "none",
                    "label_missing_reason": reason,
                    "requires_pa_manifest": target.requires_pa_manifest,
                    "n_prior_clips": 0,
                    "aggregation_method": f"{model_family}_raw_video_finetune",
                    "same_event_ensemble": False,
                    "prediction_std": None,
                    "split": sample["split"],
                }
            )
    return rows


def run_raw_video_finetune(
    base_dir: str | Path,
    *,
    clip_run_id: str = "mlb_2024_2026_full_v1",
    prediction_run_id: str = "video_raw_finetune_mlb_2024_2026_v1",
    bbe_events: str | Path | None = None,
    clips_path: str | Path | None = None,
    target_registry: str | Path = DEFAULT_TARGET_REGISTRY,
    max_clips: int | None = 500,
    num_frames: int = 16,
    image_size: int = 112,
    batch_size: int = 4,
    max_epochs: int = 5,
    learning_rate: float = 1e-4,
    model_family: str = "tiny3d",
    pretrained: bool = False,
    freeze_backbone: bool = False,
    allow_model_download: bool = False,
    hidden_dim: int = 128,
    dropout: float = 0.20,
    device: str = "auto",
    require_non_empty: bool = True,
    resume: bool = True,
    output_suffix: str = ".parquet",
    cache_dir: str | Path | None = None,
    cache_inputs: bool = False,
    cache_num_workers: int = 4,
    cache_min_free_disk_gb: float = 20.0,
    cache_max_file_mb: float | None = None,
    dataloader_num_workers: int = 0,
) -> dict[str, Path]:
    """Fine-tune a tiny 3D CNN on contact-aligned raw clips."""

    torch, _nn, _functional, DataLoader, _Dataset = _import_torch()
    base = Path(base_dir)
    outputs = {
        "predictions": base / f"predictions/{prediction_run_id}/predictions_v1{output_suffix}",
        "metrics": base / f"predictions/{prediction_run_id}/metrics_v1.json",
        "summary": base / f"reports/preflight/raw_video_finetune_{prediction_run_id}.json",
        "progress": base / f"reports/preflight/raw_video_finetune_{prediction_run_id}_progress.json",
        "checkpoint": base / f"models/video/{prediction_run_id}/checkpoint.pt",
    }
    samples, targets, input_summary = _build_samples(
        base_dir=base,
        clip_run_id=clip_run_id,
        max_clips=max_clips,
        bbe_events=bbe_events,
        clips_path=clips_path,
        target_registry=target_registry,
    )
    samples, cache_stats = _cache_samples_for_runtime(
        samples,
        cache_dir=cache_dir,
        namespace=f"runtime_io/raw_video_finetune/{prediction_run_id}/clips",
        enabled=cache_inputs,
        num_workers=cache_num_workers,
        max_file_mb=cache_max_file_mb,
        min_free_disk_gb=cache_min_free_disk_gb,
    )
    input_summary["cache_stats"] = cache_stats
    if require_non_empty and not samples:
        write_json({"schema_version": "raw_video_finetune_summary_v1", **input_summary}, outputs["summary"])
        raise RuntimeError(f"raw video fine-tune has 0 trainable samples. summary_path={outputs['summary']}")

    target_names = list(samples[0]["target_names"]) if samples else [name for name in EVENT_TARGETS if name in targets]
    train_indices = _train_indices(samples)
    normalizers = _target_normalizers(samples, target_names, targets, train_indices) if samples else {}
    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    model = _build_model(
        model_family=model_family,
        output_dim=len(target_names),
        hidden_dim=hidden_dim,
        dropout=dropout,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        allow_model_download=allow_model_download,
    ).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_history: list[float] = []
    start_epoch = 0
    resume_warning = None
    checkpoint_config = {
        "num_frames": num_frames,
        "image_size": image_size,
        "target_names": target_names,
        "model_family": model_family,
        "pretrained": pretrained,
        "freeze_backbone": freeze_backbone,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "input_samples": len(samples),
    }
    if resume and outputs["checkpoint"].exists():
        try:
            checkpoint = torch.load(outputs["checkpoint"], map_location=selected_device)
            saved_config = dict(checkpoint.get("config") or {})
            compatible = all(saved_config.get(key) == value for key, value in checkpoint_config.items())
            if compatible:
                model.load_state_dict(checkpoint["model_state"])
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                loss_history = [float(value) for value in checkpoint.get("loss_history", [])]
                start_epoch = int(checkpoint.get("epoch", -1)) + 1
            else:
                resume_warning = "existing checkpoint ignored because config/input shape changed"
        except Exception as exc:  # pragma: no cover - corrupt Colab checkpoint
            resume_warning = f"existing checkpoint ignored: {exc}"

    def write_progress(status: str, epoch: int | None = None, latest_loss: float | None = None) -> None:
        write_json(
            {
                "schema_version": "raw_video_finetune_progress_v1",
                "prediction_run_id": prediction_run_id,
                "status": status,
                "device": selected_device,
                "epoch": epoch,
                "max_epochs": max_epochs,
                "completed_epochs": len(loss_history),
                "latest_loss": latest_loss,
                "loss_history": loss_history,
                "resume": resume,
                "resume_warning": resume_warning,
                "model_family": model_family,
                "pretrained": pretrained,
                "freeze_backbone": freeze_backbone,
                "allow_model_download": allow_model_download,
                "cache_dir": None if cache_dir is None else str(cache_dir),
                "cache_inputs": cache_inputs,
                "cache_num_workers": cache_num_workers,
                "cache_min_free_disk_gb": cache_min_free_disk_gb,
                "cache_max_file_mb": cache_max_file_mb,
                "cache_stats": cache_stats,
                "dataloader_num_workers": dataloader_num_workers,
                "input_summary": input_summary,
                "outputs": {key: str(path) for key, path in outputs.items()},
            },
            outputs["progress"],
        )

    DatasetClass = _make_dataset_class(num_frames=num_frames, image_size=image_size)
    train_samples = [samples[index] for index in train_indices]
    loader_workers = max(0, int(dataloader_num_workers or 0))
    loader_kwargs = {
        "batch_size": max(1, int(batch_size)),
        "num_workers": loader_workers,
        "pin_memory": selected_device == "cuda",
    }
    if loader_workers > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(DatasetClass(train_samples), shuffle=True, **loader_kwargs)
    write_progress("started" if start_epoch == 0 else "resumed", epoch=start_epoch - 1 if start_epoch else None)
    model.train()
    for epoch in range(start_epoch, max(0, int(max_epochs))):
        epoch_losses = []
        for batch in loader:
            video = batch["video"].to(selected_device)
            y_tensor = batch["y"].to(selected_device)
            mask_tensor = batch["mask"].to(selected_device)
            y_norm = _normalize_targets(y_tensor, target_names, targets, normalizers)
            optimizer.zero_grad()
            logits = model(video)
            loss = _loss(logits, y_norm, mask_tensor, target_names, targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        latest_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        loss_history.append(latest_loss)
        outputs["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": "raw_video_finetune_checkpoint_v1",
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss_history": loss_history,
                "config": checkpoint_config,
                "target_normalizers": normalizers,
            },
            outputs["checkpoint"],
        )
        write_progress("running", epoch=epoch, latest_loss=latest_loss)

    model.eval()
    logits_by_index: dict[int, list[float]] = {}
    inference_loader = DataLoader(DatasetClass(samples), shuffle=False, **loader_kwargs)
    with torch.no_grad():
        cursor = 0
        for batch in inference_loader:
            video = batch["video"].to(selected_device)
            logits = model(video).detach().cpu().float().tolist()
            for row in logits:
                logits_by_index[cursor] = [float(value) for value in row]
                cursor += 1

    predictions = _prediction_rows(
        samples=samples,
        logits_by_index=logits_by_index,
        target_names=target_names,
        targets=targets,
        normalizers=normalizers,
        prediction_run_id=prediction_run_id,
        model_family=model_family,
    )
    validate_prediction_rows(predictions)
    metrics = evaluate_predictions(predictions, targets, run_id=prediction_run_id)
    write_table(outputs["predictions"], predictions)
    write_json(metrics, outputs["metrics"])
    write_json(
        {
            "schema_version": "raw_video_finetune_summary_v1",
            "prediction_run_id": prediction_run_id,
            "clip_run_id": clip_run_id,
            "device": selected_device,
            "input_summary": input_summary,
            "target_names": target_names,
            "train_samples": len(train_samples),
            "num_frames": num_frames,
            "image_size": image_size,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "completed_epochs": len(loss_history),
            "learning_rate": learning_rate,
            "model_family": model_family,
            "pretrained": pretrained,
            "freeze_backbone": freeze_backbone,
            "allow_model_download": allow_model_download,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "resume": resume,
            "resume_warning": resume_warning,
            "cache_dir": None if cache_dir is None else str(cache_dir),
            "cache_inputs": cache_inputs,
            "cache_num_workers": cache_num_workers,
            "cache_min_free_disk_gb": cache_min_free_disk_gb,
            "cache_max_file_mb": cache_max_file_mb,
            "cache_stats": cache_stats,
            "dataloader_num_workers": dataloader_num_workers,
            "target_normalizers": normalizers,
            "loss_history": loss_history,
            "outputs": {key: str(path) for key, path in outputs.items()},
        },
        outputs["summary"],
    )
    write_progress("complete", epoch=max(0, int(max_epochs)) - 1 if int(max_epochs) > 0 else None)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run tiny raw-video fine-tuning baseline.")
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/baseball_vision")
    parser.add_argument("--clip-run-id", default="mlb_2024_2026_full_v1")
    parser.add_argument("--prediction-run-id", default="video_raw_finetune_mlb_2024_2026_v1")
    parser.add_argument("--bbe-events", default=None)
    parser.add_argument("--clips", default=None)
    parser.add_argument("--target-registry", default=str(DEFAULT_TARGET_REGISTRY))
    parser.add_argument("--max-clips", type=int, default=500)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--model-family", choices=("tiny3d", "torchvision_r3d18"), default="tiny3d")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--output-format", choices=("parquet", "jsonl", "json", "csv"), default="parquet")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--cache-inputs", action="store_true")
    parser.add_argument("--cache-num-workers", type=int, default=4)
    parser.add_argument("--cache-min-free-disk-gb", type=float, default=20.0)
    parser.add_argument("--cache-max-file-mb", type=float, default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    args = parser.parse_args(argv)
    outputs = run_raw_video_finetune(
        args.base_dir,
        clip_run_id=args.clip_run_id,
        prediction_run_id=args.prediction_run_id,
        bbe_events=args.bbe_events,
        clips_path=args.clips,
        target_registry=args.target_registry,
        max_clips=args.max_clips,
        num_frames=args.num_frames,
        image_size=args.image_size,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        model_family=args.model_family,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        allow_model_download=args.allow_model_download,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        device=args.device,
        require_non_empty=not args.allow_empty,
        resume=not args.no_resume,
        output_suffix="." + args.output_format,
        cache_dir=args.cache_dir,
        cache_inputs=args.cache_inputs,
        cache_num_workers=args.cache_num_workers,
        cache_min_free_disk_gb=args.cache_min_free_disk_gb,
        cache_max_file_mb=args.cache_max_file_mb,
        dataloader_num_workers=args.dataloader_num_workers,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
