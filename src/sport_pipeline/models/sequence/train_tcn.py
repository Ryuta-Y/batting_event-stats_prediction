"""Colab trainable TCN/MS-TCN structured-sequence baseline.

This is the first real trainable model for Agent C outputs. It consumes
structured_sequence_v1 frame artifacts, trains event-level heads from the target
registry, masks optional xBA/xwOBA labels, and never emits OPS as an event head.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

from sport_pipeline.artifact_check import write_json
from sport_pipeline.evaluation import evaluate_predictions, load_target_registry, validate_prediction_rows
from sport_pipeline.io import read_table, write_table


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


def _target_missing_reason(event: dict[str, Any], target_name: str) -> str:
    return str(
        event.get(f"target_{target_name}_missing_reason")
        or event.get("label_missing_reason")
        or "label_missing"
    )


def _resolve_input_paths(
    base: Path,
    sequence_manifest: str | Path | None,
    frames: str | Path | None,
    bbe_events: str | Path | None,
    event_with_prior: str | Path | None,
) -> tuple[Path, Path, Path, Path]:
    sequence_file = Path(sequence_manifest) if sequence_manifest else base / "features/structured_sequence_v1/manifest.parquet"
    frame_file = Path(frames) if frames else base / "features/structured_sequence_v1/frames.parquet"
    bbe_file = Path(bbe_events) if bbe_events else base / "manifests/bbe_events_v1.parquet"
    prior_file = Path(event_with_prior) if event_with_prior else base / "datasets/event_with_player_prior_v1/manifest.parquet"
    return sequence_file, frame_file, bbe_file, prior_file


def _read_existing_table(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_table(path)


def summarize_tcn_inputs(
    base_dir: str | Path,
    *,
    sequence_manifest: str | Path | None = None,
    frames: str | Path | None = None,
    bbe_events: str | Path | None = None,
    event_with_prior: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize TCN input artifacts before importing torch or training."""

    base = Path(base_dir)
    sequence_file, frame_file, bbe_file, prior_file = _resolve_input_paths(
        base,
        sequence_manifest,
        frames,
        bbe_events,
        event_with_prior,
    )
    manifest_rows = _read_existing_table(sequence_file)
    frame_rows = _read_existing_table(frame_file)
    bbe_rows = _read_existing_table(bbe_file)
    prior_rows = _read_existing_table(prior_file)
    frame_sequence_ids = {str(row.get("sequence_id")) for row in frame_rows}
    event_ids = {str(row.get("event_id")) for row in bbe_rows}
    sequences_with_frames = [row for row in manifest_rows if str(row.get("sequence_id")) in frame_sequence_ids]
    sequences_with_events = [row for row in manifest_rows if str(row.get("event_id")) in event_ids]
    matched_sequences = [
        row
        for row in manifest_rows
        if str(row.get("sequence_id")) in frame_sequence_ids and str(row.get("event_id")) in event_ids
    ]
    reason = None
    if not sequence_file.exists():
        reason = "missing structured sequence manifest; run notebooks/17_deep_sequence_features.ipynb after clips exist"
    elif not frame_file.exists():
        reason = "missing structured sequence frames; run notebooks/17_deep_sequence_features.ipynb"
    elif not bbe_file.exists():
        reason = "missing bbe_events_v1; run notebooks/02_build_manifest.ipynb or 11 with Statcast download enabled"
    elif not manifest_rows:
        reason = "structured sequence manifest has 0 rows; clips_v1 is probably empty or 17 overwrote features with no clips"
    elif not frame_rows:
        reason = "structured sequence frames has 0 rows; rerun 17 after non-empty clips_v1 exists"
    elif not bbe_rows:
        reason = "bbe_events_v1 has 0 rows; rebuild Statcast BBE manifest"
    elif not matched_sequences:
        reason = "no sequence rows have both frame rows and matching BBE event_id"
    return {
        "paths": {
            "sequence_manifest": str(sequence_file),
            "frames": str(frame_file),
            "bbe_events": str(bbe_file),
            "event_with_prior": str(prior_file),
        },
        "exists": {
            "sequence_manifest": sequence_file.exists(),
            "frames": frame_file.exists(),
            "bbe_events": bbe_file.exists(),
            "event_with_prior": prior_file.exists(),
        },
        "row_counts": {
            "sequence_manifest": len(manifest_rows),
            "frames": len(frame_rows),
            "bbe_events": len(bbe_rows),
            "event_with_prior": len(prior_rows),
            "sequences_with_frames": len(sequences_with_frames),
            "sequences_with_events": len(sequences_with_events),
            "matched_trainable_sequences": len(matched_sequences),
        },
        "examples": {
            "sequence_ids_without_frames": [
                str(row.get("sequence_id"))
                for row in manifest_rows
                if str(row.get("sequence_id")) not in frame_sequence_ids
            ][:10],
            "event_ids_without_bbe": [
                str(row.get("event_id"))
                for row in manifest_rows
                if str(row.get("event_id")) not in event_ids
            ][:10],
        },
        "likely_stop_reason": reason,
        "next_action_ja": (
            "まず 12 の clips_v1 が non-empty か確認し、13 または 17 を再実行して "
            "features/structured_sequence_v1/{manifest,frames}.parquet に行が入ってから 18 を実行してください。"
        ),
    }


def _contact_label_from_frame(frame: dict[str, Any]) -> float:
    feature_names = frame.get("feature_names") or []
    feature_values = frame.get("feature_values") or []
    if isinstance(feature_names, list) and "contact_phase_score" in feature_names:
        index = feature_names.index("contact_phase_score")
        if isinstance(feature_values, list) and index < len(feature_values):
            return _to_float(feature_values[index], 0.0)
    return 1.0 if str(frame.get("phase_label") or "") == "contact" else 0.0


def _feature_value_from_frame(frame: dict[str, Any], feature_name: str, default: float = 0.0) -> float:
    if frame.get(feature_name) is not None:
        return _to_float(frame.get(feature_name), default)
    feature_names = frame.get("feature_names") or []
    feature_values = frame.get("feature_values") or []
    if isinstance(feature_names, list) and isinstance(feature_values, list) and feature_name in feature_names:
        index = feature_names.index(feature_name)
        if index < len(feature_values):
            return _to_float(feature_values[index], default)
    return default


def _load_samples(
    base: Path,
    sequence_manifest_path: Path,
    frame_path: Path,
    bbe_path: Path,
    event_with_prior_path: Path | None = None,
) -> list[dict[str, Any]]:
    manifest_rows = read_table(sequence_manifest_path)
    frame_rows = read_table(frame_path)
    bbe_rows = read_table(bbe_path)
    events = {str(row["event_id"]): row for row in bbe_rows}
    prior_rows = read_table(event_with_prior_path) if event_with_prior_path is not None and event_with_prior_path.exists() else []
    prior_by_event = {str(row["event_id"]): row for row in prior_rows}
    frames_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        frames_by_sequence[str(row["sequence_id"])].append(row)
    samples: list[dict[str, Any]] = []
    for sequence in manifest_rows:
        sequence_id = str(sequence["sequence_id"])
        frames = sorted(frames_by_sequence.get(sequence_id, []), key=lambda row: int(row["frame_index"]))
        event = events.get(str(sequence["event_id"]))
        if not frames or event is None:
            continue
        feature_values = [list(map(float, frame["feature_values"])) for frame in frames]
        contact_labels = [_contact_label_from_frame(frame) for frame in frames]
        samples.append(
            {
                "sample_id": str(sequence["sample_id"]),
                "sequence_id": sequence_id,
                "event_id": str(sequence["event_id"]),
                "batter_season_id": str(sequence["batter_season_id"]),
                "split": str(sequence.get("split") or "unknown"),
                "features": feature_values,
                "contact_labels": contact_labels,
                "event": event,
                "prior": prior_by_event.get(str(sequence["event_id"])),
            }
        )
    return samples


def _import_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        import torch.nn.functional as functional  # type: ignore

        return torch, nn, functional
    except ImportError as exc:  # pragma: no cover - Colab dependency path
        raise RuntimeError("PyTorch is required for train_tcn. Use a Colab GPU runtime.") from exc


def _build_model(
    input_dim: int,
    output_dim: int,
    model_family: str,
    hidden_dim: int,
    depth: int,
    kernel_size: int,
    dropout: float,
):
    torch, nn, _functional = _import_torch()

    class TemporalBlock(nn.Module):
        def __init__(self, channels: int, dilation: int) -> None:
            super().__init__()
            padding = dilation * (kernel_size - 1) // 2
            self.net = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
                nn.Dropout(float(dropout)),
            )
            self.activation = nn.ReLU()

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.activation(x + self.net(x))

    class TCNRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            stages = 2 if model_family == "ms_tcn" else 1
            layers = [nn.Conv1d(input_dim, hidden_dim, kernel_size=1), nn.ReLU()]
            for _stage in range(stages):
                for layer_index in range(depth):
                    layers.append(TemporalBlock(hidden_dim, dilation=2**layer_index))
            self.encoder = nn.Sequential(*layers)
            self.event_head = nn.Linear(hidden_dim, output_dim)
            self.contact_head = nn.Linear(hidden_dim, 1)

        def forward(self, x, frame_mask=None):  # type: ignore[no-untyped-def]
            # x: N, T, D
            encoded = self.encoder(x.transpose(1, 2))
            if frame_mask is None:
                pooled = encoded.mean(dim=2)
            else:
                mask = frame_mask.unsqueeze(1).to(encoded.dtype)
                pooled = (encoded * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
            contact_logits = self.contact_head(encoded.transpose(1, 2)).squeeze(-1)
            return self.event_head(pooled), contact_logits

    return TCNRegressor()


def _prepare_tensors(samples: list[dict[str, Any]], targets: dict[str, Any], prior_feature_mode: str):
    torch, _nn, _functional = _import_torch()
    if not samples:
        raise ValueError("no structured sequence samples available")
    max_t = max(len(sample["features"]) for sample in samples)
    dim = len(samples[0]["features"][0])
    prior_dim = 0
    if prior_feature_mode != "none":
        prior_dim = max(
            (len((sample.get("prior") or {}).get("player_season_embedding_values") or []) for sample in samples),
            default=0,
        )
    x_values = []
    contact_values = []
    frame_mask_values = []
    for sample in samples:
        rows = sample["features"]
        if any(len(row) != dim for row in rows):
            raise ValueError("all feature rows must share one feature_dim")
        prior_values = []
        if prior_dim:
            raw_prior = (sample.get("prior") or {}).get("player_season_embedding_values") or []
            prior_values = [float(value) for value in raw_prior[:prior_dim]]
            prior_values = prior_values + [0.0] * (prior_dim - len(prior_values))
        conditioned_rows = [row + prior_values for row in rows]
        padded = conditioned_rows + [[0.0] * (dim + prior_dim) for _ in range(max_t - len(rows))]
        x_values.append(padded)
        raw_contact = [float(value) for value in sample.get("contact_labels", [])[: len(rows)]]
        contact_values.append(raw_contact + [0.0] * (max_t - len(raw_contact)))
        frame_mask_values.append([1.0] * len(rows) + [0.0] * (max_t - len(rows)))
    target_names = [name for name in EVENT_TARGETS if name in targets and targets[name].level == "event"]
    if "ops" in target_names:
        raise ValueError("OPS must not be trained as an event-level sequence head")
    y_values: list[list[float]] = []
    mask_values: list[list[float]] = []
    for sample in samples:
        event = sample["event"]
        y_row: list[float] = []
        mask_row: list[float] = []
        for target_name in target_names:
            target = targets[target_name]
            value = event.get(target.column)
            if _is_missing(value):
                y_row.append(0.0)
                mask_row.append(0.0)
            else:
                y_row.append(float(value))
                mask_row.append(1.0)
        y_values.append(y_row)
        mask_values.append(mask_row)
    return (
        torch.tensor(x_values, dtype=torch.float32),
        torch.tensor(y_values, dtype=torch.float32),
        torch.tensor(mask_values, dtype=torch.float32),
        torch.tensor(contact_values, dtype=torch.float32),
        torch.tensor(frame_mask_values, dtype=torch.float32),
        target_names,
    )


def _target_normalizers(
    y_tensor: Any,
    mask_tensor: Any,
    train_index_tensor: Any,
    target_names: list[str],
    target_specs: dict[str, Any],
) -> dict[str, dict[str, float | str]]:
    torch, _nn, _functional = _import_torch()
    normalizers: dict[str, dict[str, float | str]] = {}
    for index, target_name in enumerate(target_names):
        target = target_specs[target_name]
        if target.kind in {"binary", "probability"}:
            normalizers[target_name] = {"kind": target.kind, "mean": 0.0, "std": 1.0}
            continue
        train_mask = mask_tensor[train_index_tensor, index] > 0
        if not torch.any(train_mask):
            normalizers[target_name] = {"kind": target.kind, "mean": 0.0, "std": 1.0}
            continue
        values = y_tensor[train_index_tensor, index][train_mask]
        mean_value = values.mean()
        std_value = values.std(unbiased=False).clamp_min(1.0e-6)
        normalizers[target_name] = {
            "kind": target.kind,
            "mean": float(mean_value.detach().cpu()),
            "std": float(std_value.detach().cpu()),
        }
    return normalizers


def _normalize_targets(y_tensor: Any, target_names: list[str], target_specs: dict[str, Any], normalizers: dict[str, dict[str, float | str]]) -> Any:
    y_normalized = y_tensor.clone()
    for index, target_name in enumerate(target_names):
        if target_specs[target_name].kind in {"binary", "probability"}:
            continue
        normalizer = normalizers[target_name]
        y_normalized[:, index] = (y_normalized[:, index] - float(normalizer["mean"])) / float(normalizer["std"])
    return y_normalized


def _inverse_normalize_prediction(
    value: float,
    target_name: str,
    target_specs: dict[str, Any],
    normalizers: dict[str, dict[str, float | str]],
) -> float:
    if target_specs[target_name].kind in {"binary", "probability"}:
        return value
    normalizer = normalizers[target_name]
    return value * float(normalizer["std"]) + float(normalizer["mean"])


def _masked_loss(logits: Any, targets_tensor: Any, masks: Any, target_names: list[str], target_specs: dict[str, Any]) -> Any:
    torch, _nn, functional = _import_torch()
    losses = []
    for index, target_name in enumerate(target_names):
        mask = masks[:, index] > 0
        if not torch.any(mask):
            continue
        pred = logits[mask, index]
        truth = targets_tensor[mask, index]
        if target_specs[target_name].kind in {"binary", "probability"}:
            losses.append(functional.binary_cross_entropy_with_logits(pred, truth))
        elif target_specs[target_name].loss == "huber":
            losses.append(functional.smooth_l1_loss(pred, truth))
        else:
            losses.append(functional.mse_loss(pred, truth))
    if not losses:
        return logits.sum() * 0.0
    return sum(losses) / len(losses)


def _contact_aux_loss(contact_logits: Any, contact_targets: Any, frame_masks: Any) -> Any:
    torch, _nn, functional = _import_torch()
    valid = frame_masks > 0
    if not torch.any(valid):
        return contact_logits.sum() * 0.0
    return functional.binary_cross_entropy_with_logits(contact_logits[valid], contact_targets[valid].clamp(0.0, 1.0))


def run_tcn_training(
    base_dir: str | Path,
    *,
    prediction_run_id: str = "sequence_tcn_mlb_2024_2026_v1",
    sequence_manifest: str | Path | None = None,
    frames: str | Path | None = None,
    bbe_events: str | Path | None = None,
    event_with_prior: str | Path | None = None,
    target_registry: str | Path = DEFAULT_TARGET_REGISTRY,
    prior_feature_mode: str = "concat_if_available",
    model_family: str = "tcn",
    hidden_dim: int = 64,
    depth: int = 3,
    kernel_size: int = 3,
    dropout: float = 0.10,
    contact_aux_weight: float = 0.20,
    max_epochs: int = 20,
    learning_rate: float = 1e-3,
    device: str = "auto",
    output_suffix: str = ".parquet",
    resume: bool = True,
    checkpoint_every_epoch: bool = True,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Path]:
    """Train a TCN/MS-TCN on structured sequence features and write predictions_v1."""

    base = Path(base_dir)
    outputs = {
        "predictions": base / f"predictions/{prediction_run_id}/predictions_v1{output_suffix}",
        "metrics": base / f"predictions/{prediction_run_id}/metrics_v1.json",
        "summary": base / f"reports/preflight/train_tcn_{prediction_run_id}.json",
        "progress": base / f"reports/preflight/train_tcn_{prediction_run_id}_progress.json",
        "checkpoint": Path(checkpoint_path)
        if checkpoint_path is not None
        else base / f"models/sequence/{prediction_run_id}/checkpoint.pt",
    }
    sequence_file, frame_file, bbe_file, prior_file = _resolve_input_paths(
        base,
        sequence_manifest,
        frames,
        bbe_events,
        event_with_prior,
    )
    targets = load_target_registry(target_registry)
    samples = _load_samples(base, sequence_file, frame_file, bbe_file, prior_file)
    if not samples:
        input_summary = summarize_tcn_inputs(
            base,
            sequence_manifest=sequence_file,
            frames=frame_file,
            bbe_events=bbe_file,
            event_with_prior=prior_file,
        )
        raise ValueError(
            "no structured sequence samples available; "
            f"input_summary={json.dumps(input_summary, ensure_ascii=False, sort_keys=True)}"
        )
    input_frame_rows = read_table(frame_file) if frame_file.exists() else []
    input_pose_coverage_rows = sum(
        1 for row in input_frame_rows if _feature_value_from_frame(row, "pose_coverage", 0.0) > 0.0
    )
    torch, _nn, _functional = _import_torch()
    x_tensor, y_tensor, mask_tensor, contact_tensor, frame_mask_tensor, target_names = _prepare_tensors(samples, targets, prior_feature_mode)
    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    x_tensor = x_tensor.to(selected_device)
    y_tensor = y_tensor.to(selected_device)
    mask_tensor = mask_tensor.to(selected_device)
    contact_tensor = contact_tensor.to(selected_device)
    frame_mask_tensor = frame_mask_tensor.to(selected_device)

    train_indices = [idx for idx, sample in enumerate(samples) if sample["split"] == "train"]
    if not train_indices:
        train_indices = list(range(len(samples)))
    train_index_tensor = torch.tensor(train_indices, dtype=torch.long, device=selected_device)
    target_normalizers = _target_normalizers(y_tensor, mask_tensor, train_index_tensor, target_names, targets)
    normalized_y_tensor = _normalize_targets(y_tensor, target_names, targets, target_normalizers)
    model = _build_model(
        input_dim=x_tensor.shape[-1],
        output_dim=len(target_names),
        model_family=model_family,
        hidden_dim=hidden_dim,
        depth=depth,
        kernel_size=kernel_size,
        dropout=dropout,
    ).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_history: list[float] = []
    checkpoint_config = {
        "model_family": model_family,
        "input_dim": int(x_tensor.shape[-1]),
        "output_dim": len(target_names),
        "target_names": target_names,
        "hidden_dim": hidden_dim,
        "depth": depth,
        "kernel_size": kernel_size,
        "dropout": dropout,
        "prior_feature_mode": prior_feature_mode,
        "input_samples": len(samples),
        "train_samples": len(train_indices),
    }
    resume_warning = None
    start_epoch = 0
    checkpoint_file = outputs["checkpoint"]
    if resume and checkpoint_file.exists():
        try:
            checkpoint = torch.load(checkpoint_file, map_location=selected_device)
            saved_config = dict(checkpoint.get("config") or {})
            compatible = all(saved_config.get(key) == value for key, value in checkpoint_config.items())
            if compatible:
                model.load_state_dict(checkpoint["model_state"])
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                loss_history = [float(value) for value in checkpoint.get("loss_history", [])]
                start_epoch = int(checkpoint.get("epoch", -1)) + 1
            else:
                resume_warning = "existing checkpoint ignored because config/input shape changed"
        except Exception as exc:  # pragma: no cover - corrupt runtime checkpoint
            resume_warning = f"existing checkpoint ignored: {exc}"

    def write_training_progress(status: str, epoch: int | None = None, latest_loss: float | None = None) -> None:
        write_json(
            {
                "schema_version": "sequence_tcn_training_progress_v1",
                "prediction_run_id": prediction_run_id,
                "status": status,
                "device": selected_device,
                "model_family": model_family,
                "max_epochs": max_epochs,
                "start_epoch": start_epoch,
                "last_epoch": epoch,
                "completed_epochs": len(loss_history),
                "latest_loss": latest_loss,
                "loss_history": loss_history,
                "checkpoint_path": str(checkpoint_file),
                "resume": resume,
                "checkpoint_every_epoch": checkpoint_every_epoch,
                "resume_warning": resume_warning,
                "input_samples": len(samples),
                "input_pose_coverage_rows": input_pose_coverage_rows,
                "train_samples": len(train_indices),
                "target_names": target_names,
                "output_predictions": str(outputs["predictions"]),
                "output_metrics": str(outputs["metrics"]),
            },
            outputs["progress"],
        )

    write_training_progress("started" if start_epoch == 0 else "resumed", epoch=start_epoch - 1 if start_epoch else None)
    model.train()
    for epoch in range(start_epoch, max(0, int(max_epochs))):
        optimizer.zero_grad()
        logits, contact_logits = model(x_tensor[train_index_tensor], frame_mask_tensor[train_index_tensor])
        event_loss = _masked_loss(logits, normalized_y_tensor[train_index_tensor], mask_tensor[train_index_tensor], target_names, targets)
        contact_loss = _contact_aux_loss(contact_logits, contact_tensor[train_index_tensor], frame_mask_tensor[train_index_tensor])
        loss = event_loss + float(contact_aux_weight) * contact_loss
        loss.backward()
        optimizer.step()
        latest_loss = float(loss.detach().cpu())
        loss_history.append(latest_loss)
        if checkpoint_every_epoch:
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema_version": "sequence_tcn_checkpoint_v1",
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss_history": loss_history,
                    "config": checkpoint_config,
                    "target_normalizers": target_normalizers,
                },
                checkpoint_file,
            )
        write_training_progress("running", epoch=epoch, latest_loss=latest_loss)
    write_training_progress("training_complete", epoch=max(0, int(max_epochs)) - 1 if int(max_epochs) > 0 else None)

    model.eval()
    with torch.no_grad():
        logits, _contact_logits = model(x_tensor, frame_mask_tensor)
        logits = logits.detach().cpu()

    predictions: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        event = sample["event"]
        for target_index, target_name in enumerate(target_names):
            target = targets[target_name]
            value = event.get(target.column)
            trained_for_target = bool(mask_tensor[train_index_tensor, target_index].sum().detach().cpu().item() > 0)
            available = not _is_missing(value) and trained_for_target
            y_pred = None
            missing_reason = None
            if available:
                raw_pred = float(logits[sample_index, target_index])
                if target.kind in {"binary", "probability"}:
                    y_pred = float(1.0 / (1.0 + math.exp(-raw_pred)))
                else:
                    y_pred = _inverse_normalize_prediction(raw_pred, target_name, targets, target_normalizers)
            elif _is_missing(value):
                missing_reason = _target_missing_reason(event, target_name)
            else:
                missing_reason = "sequence_tcn_not_fit_for_target"
            predictions.append(
                {
                    "run_id": prediction_run_id,
                    "sample_id": sample["sample_id"],
                    "event_id": sample["event_id"],
                    "batter_season_id": sample["batter_season_id"],
                    "prediction_level": "event",
                    "target_name": target_name,
                    "y_true": None if _is_missing(value) else float(value),
                    "y_pred": y_pred,
                    "target_available": available,
                    "target_source": target.column,
                    "head_kind": target.kind,
                    "loss_name": target.loss,
                    "aggregation_scope": str(
                        (sample.get("prior") or {}).get("aggregation_scope")
                        or "current_event_structured_sequence"
                    ),
                    "prior_mode": str((sample.get("prior") or {}).get("prior_mode") or "none"),
                    "label_missing_reason": missing_reason,
                    "requires_pa_manifest": target.requires_pa_manifest,
                    "n_prior_clips": int((sample.get("prior") or {}).get("n_prior_clips") or 0),
                    "aggregation_method": str((sample.get("prior") or {}).get("aggregation_method") or model_family),
                    "same_event_ensemble": False,
                    "prediction_std": None,
                    "split": sample["split"],
                }
            )

    validate_prediction_rows(predictions)
    metrics = evaluate_predictions(predictions, targets, run_id=prediction_run_id)
    write_table(outputs["predictions"], predictions)
    write_json(metrics, outputs["metrics"])
    write_json(
        {
            "schema_version": "sequence_tcn_training_summary_v1",
            "prediction_run_id": prediction_run_id,
            "model_family": model_family,
            "device": selected_device,
            "input_samples": len(samples),
            "input_pose_coverage_rows": input_pose_coverage_rows,
            "train_samples": len(train_indices),
            "target_names": target_names,
            "event_with_prior_path": str(prior_file),
            "prior_feature_mode": prior_feature_mode,
            "hidden_dim": hidden_dim,
            "depth": depth,
            "kernel_size": kernel_size,
            "dropout": dropout,
            "contact_aux_weight": contact_aux_weight,
            "max_epochs": max_epochs,
            "completed_epochs": len(loss_history),
            "resume": resume,
            "resumed_from_epoch": start_epoch if start_epoch > 0 else None,
            "checkpoint_every_epoch": checkpoint_every_epoch,
            "checkpoint_path": str(checkpoint_file),
            "progress_path": str(outputs["progress"]),
            "resume_warning": resume_warning,
            "target_normalizers": target_normalizers,
            "loss_history": loss_history,
            "note": "This is Colab/GPU intended; do not run large training locally.",
        },
        outputs["summary"],
    )
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train TCN/MS-TCN structured-sequence baseline.")
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/baseball_vision")
    parser.add_argument("--prediction-run-id", default="sequence_tcn_mlb_2024_2026_v1")
    parser.add_argument("--sequence-manifest", default=None)
    parser.add_argument("--frames", default=None)
    parser.add_argument("--bbe-events", default=None)
    parser.add_argument("--event-with-prior", default=None)
    parser.add_argument("--target-registry", default=str(DEFAULT_TARGET_REGISTRY))
    parser.add_argument("--prior-feature-mode", choices=("none", "concat_if_available"), default="concat_if_available")
    parser.add_argument("--model-family", choices=("tcn", "ms_tcn"), default="tcn")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--contact-aux-weight", type=float, default=0.20)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-format", choices=("parquet", "jsonl", "json", "csv"), default="parquet")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-path", default=None)
    args = parser.parse_args(argv)
    outputs = run_tcn_training(
        args.base_dir,
        prediction_run_id=args.prediction_run_id,
        sequence_manifest=args.sequence_manifest,
        frames=args.frames,
        bbe_events=args.bbe_events,
        event_with_prior=args.event_with_prior,
        target_registry=args.target_registry,
        prior_feature_mode=args.prior_feature_mode,
        model_family=args.model_family,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        contact_aux_weight=args.contact_aux_weight,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        output_suffix="." + args.output_format,
        resume=not args.no_resume,
        checkpoint_every_epoch=not args.no_checkpoint,
        checkpoint_path=args.checkpoint_path,
    )
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
