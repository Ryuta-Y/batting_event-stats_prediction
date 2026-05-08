"""Fill VLM mechanics manifests with Hugging Face open-source VLM outputs.

The primary supported path is Qwen2.5-VL on Colab GPU. This module intentionally
keeps model execution opt-in because VLM weights are large and may require a
Hugging Face token / license acceptance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

from sport_pipeline.artifact_check import write_json
from sport_pipeline.io import read_table, write_table
from sport_pipeline.io.runtime_cache import cache_file


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
JSON_OBJECT_COLUMNS = ("vlm_labels", "numeric_features")


MECHANICS_JSON_PROMPT = """Analyze this baseball batting swing clip.
Return only valid JSON with these keys:
{
  "mechanics_caption": "one concise sentence",
  "stance": "open|closed|neutral|unknown",
  "load": "early|on_time|late|unknown",
  "stride": "short|medium|long|unknown",
  "bat_path": "level|uppercut|steep|unknown",
  "contact_timing": "early|on_time|late|unknown",
  "balance": "balanced|off_balance|unknown",
  "hip_rotation_score": 0.0,
  "bat_path_steepness_score": 0.0,
  "timing_score": 0.0,
  "balance_score": 0.0
}
Scores must be between 0 and 1. Do not infer Statcast outcome labels directly."""


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


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


def _serialise_json_object_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _write_vlm_feature_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    return write_table(path, _serialise_json_object_columns(rows))


def _json_object_from_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        payload = json.loads(stripped)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(stripped[start : end + 1])
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _score(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(output) or math.isinf(output):
        return None
    return min(max(output, 0.0), 1.0)


def normalise_vlm_mechanics_output(text: str) -> tuple[str, dict[str, Any], dict[str, float]]:
    """Parse a VLM response into caption, categorical labels, and numeric scores."""

    payload = _json_object_from_text(text)
    caption = str(payload.get("mechanics_caption") or payload.get("caption") or text).strip()
    labels = {
        key: str(payload.get(key) or "unknown")
        for key in ("stance", "load", "stride", "bat_path", "contact_timing", "balance")
    }
    numeric = {}
    for key in ("hip_rotation_score", "bat_path_steepness_score", "timing_score", "balance_score"):
        value = _score(payload.get(key))
        if value is not None:
            numeric[key] = value
    return caption, labels, numeric


def _configure_qwen_video_reader(video_reader_backend: str | None) -> None:
    if not video_reader_backend:
        return
    os.environ["FORCE_QWENVL_VIDEO_READER"] = video_reader_backend
    module = sys.modules.get("qwen_vl_utils.vision_process")
    if module is not None:
        setattr(module, "FORCE_QWENVL_VIDEO_READER", video_reader_backend)
        getter = getattr(module, "get_video_reader_backend", None)
        cache_clear = getattr(getter, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()


def _load_qwen_stack(model_id: str, *, token: str | None, dtype: str, trust_remote_code: bool):
    try:
        import torch  # type: ignore
        from transformers import AutoProcessor  # type: ignore
        try:
            from transformers import AutoModelForImageTextToText as AutoModelClass  # type: ignore
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoModelClass  # type: ignore
        from qwen_vl_utils import process_vision_info  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "HF VLM captioning requires torch, transformers, accelerate, and qwen-vl-utils in Colab. "
            "Install them in notebook 24b before running captioning."
        ) from exc

    torch_dtype = "auto"
    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    processor = AutoProcessor.from_pretrained(model_id, token=token, trust_remote_code=trust_remote_code)
    model = AutoModelClass.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        token=token,
        trust_remote_code=trust_remote_code,
    )
    return torch, processor, model, process_vision_info


def _bytes_from_mb(value: int | float | None) -> int | None:
    if value is None:
        return None
    return max(0, int(float(value) * 1024**2))


def _bytes_from_gb(value: int | float | None, default_gb: float = 20.0) -> int:
    if value is None:
        value = default_gb
    return max(0, int(float(value) * 1024**3))


def _cache_media_for_row(
    row: dict[str, Any],
    *,
    cache_dir: str | Path | None,
    namespace: str,
    input_mode: str,
    enabled: bool,
    max_file_mb: float | None,
    min_free_disk_gb: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stats = {"enabled": bool(enabled and cache_dir is not None), "used": False, "reason": "cache_disabled"}
    if not enabled or cache_dir is None:
        return row, stats
    path_key = "debug_frame_path" if input_mode == "debug_frame" else "clip_path"
    source = row.get(path_key)
    if not source:
        stats["reason"] = "source_missing"
        return row, stats
    result = cache_file(
        source,
        cache_dir=cache_dir,
        namespace=namespace,
        key=str(row.get("clip_id") or row.get("sample_id") or ""),
        enabled=True,
        max_file_bytes=_bytes_from_mb(max_file_mb),
        min_free_disk_bytes=_bytes_from_gb(min_free_disk_gb),
    )
    output = dict(row)
    runtime_key = "_runtime_debug_frame_path" if input_mode == "debug_frame" else "_runtime_clip_path"
    output[runtime_key] = str(result.path)
    stats["used"] = result.used_cache
    stats["reason"] = result.reason
    return output, stats


def _message_for_row(row: dict[str, Any], *, input_mode: str, fps: float, max_pixels: int | None) -> list[dict[str, Any]]:
    clip_path = row.get("_runtime_clip_path") or row.get("clip_path")
    debug_frame_path = row.get("_runtime_debug_frame_path") or row.get("debug_frame_path")
    content: list[dict[str, Any]] = []
    if input_mode == "debug_frame":
        if not debug_frame_path or not Path(str(debug_frame_path)).exists():
            raise FileNotFoundError(f"debug_frame_path is missing for clip_id={row.get('clip_id')}")
        image_item: dict[str, Any] = {"type": "image", "image": Path(str(debug_frame_path)).resolve().as_uri()}
        if max_pixels:
            image_item["max_pixels"] = max_pixels
        content.append(image_item)
    else:
        if not clip_path or not Path(str(clip_path)).exists():
            raise FileNotFoundError(f"clip_path is missing for clip_id={row.get('clip_id')}")
        video_item: dict[str, Any] = {"type": "video", "video": Path(str(clip_path)).resolve().as_uri(), "fps": fps}
        if max_pixels:
            video_item["max_pixels"] = max_pixels
        content.append(video_item)
    content.append({"type": "text", "text": str(row.get("vlm_prompt") or MECHANICS_JSON_PROMPT)})
    return [{"role": "user", "content": content}]


def _processor_media_kwargs(
    image_inputs: Any,
    video_inputs: Any,
    video_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if image_inputs:
        kwargs["images"] = image_inputs
    if not video_inputs:
        return kwargs
    kwargs["videos"] = video_inputs
    for key, value in (video_kwargs or {}).items():
        if key == "fps" and isinstance(value, (list, tuple)):
            if len(value) == 0:
                continue
            if len(value) == 1:
                value = value[0]
        if isinstance(value, (list, tuple)) and len(value) == 0:
            continue
        kwargs[key] = value
    return kwargs


def _run_vlm_generation(
    row: dict[str, Any],
    *,
    input_mode: str,
    fps: float,
    max_pixels: int | None,
    processor: Any,
    model: Any,
    process_vision_info: Any,
) -> tuple[str, dict[str, Any], dict[str, float], str]:
    messages = _message_for_row(row, input_mode=input_mode, fps=fps, max_pixels=max_pixels)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    media_kwargs = _processor_media_kwargs(image_inputs, video_inputs, video_kwargs)
    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
        **media_kwargs,
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    caption, labels, numeric = normalise_vlm_mechanics_output(response)
    return caption, labels, numeric, response


def fill_vlm_manifest_with_hf(
    base_dir: str | Path,
    *,
    vlm_feature_id: str = "vlm_mechanics_v1",
    model_id: str = DEFAULT_MODEL_ID,
    input_mode: str = "clip_video",
    max_rows: int | None = None,
    fps: float = 1.0,
    max_pixels: int | None = 360 * 420,
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    video_reader_backend: str | None = None,
    fallback_to_debug_frame: bool = True,
    token_env: str = "HF_TOKEN",
    output_suffix: str = ".parquet",
    cache_dir: str | Path | None = None,
    cache_inputs: bool = False,
    cache_min_free_disk_gb: float = 20.0,
    cache_max_file_mb: float | None = None,
) -> dict[str, Path]:
    """Run an HF VLM over rows whose feature_status is not complete."""

    if input_mode not in {"clip_video", "debug_frame"}:
        raise ValueError("input_mode must be clip_video or debug_frame")
    base = Path(base_dir)
    manifest = base / f"features/{vlm_feature_id}/manifest{output_suffix}"
    summary = base / f"reports/preflight/hf_vlm_captioning_{vlm_feature_id}.json"
    rows = _read_vlm_feature_rows(manifest) if manifest.exists() else []
    token = os.environ.get(token_env) or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    _configure_qwen_video_reader(video_reader_backend)
    _torch, processor, model, process_vision_info = _load_qwen_stack(
        model_id,
        token=token,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )

    updated_rows: list[dict[str, Any]] = []
    processed = 0
    failures: list[dict[str, Any]] = []
    initial_complete_rows = sum(
        1
        for row in rows
        if bool(row.get("vlm_caption")) or row.get("feature_status") == "vlm_complete"
    )
    cache_stats: dict[str, Any] = {"enabled": bool(cache_inputs and cache_dir is not None), "used": 0, "reasons": {}}
    for row in rows:
        output = dict(row)
        already_done = bool(output.get("vlm_caption")) or output.get("feature_status") == "vlm_complete"
        complete_rows_so_far = initial_complete_rows + processed
        if already_done or (max_rows is not None and complete_rows_so_far >= max_rows):
            updated_rows.append(output)
            continue
        try:
            output, media_cache_stats = _cache_media_for_row(
                output,
                cache_dir=cache_dir,
                namespace=f"runtime_io/hf_vlm_captioning/{vlm_feature_id}/media",
                input_mode=input_mode,
                enabled=cache_inputs,
                max_file_mb=cache_max_file_mb,
                min_free_disk_gb=cache_min_free_disk_gb,
            )
            reason = str(media_cache_stats.get("reason") or "unknown")
            cache_stats["reasons"][reason] = int(cache_stats["reasons"].get(reason, 0)) + 1
            if media_cache_stats.get("used"):
                cache_stats["used"] += 1
            used_input_mode = input_mode
            try:
                caption, labels, numeric, response = _run_vlm_generation(
                    output,
                    input_mode=input_mode,
                    fps=fps,
                    max_pixels=max_pixels,
                    processor=processor,
                    model=model,
                    process_vision_info=process_vision_info,
                )
            except Exception as video_exc:
                if input_mode != "clip_video" or not fallback_to_debug_frame:
                    raise
                fallback_row, fallback_cache_stats = _cache_media_for_row(
                    output,
                    cache_dir=cache_dir,
                    namespace=f"runtime_io/hf_vlm_captioning/{vlm_feature_id}/debug_frames",
                    input_mode="debug_frame",
                    enabled=cache_inputs,
                    max_file_mb=cache_max_file_mb,
                    min_free_disk_gb=cache_min_free_disk_gb,
                )
                fallback_reason = str(fallback_cache_stats.get("reason") or "unknown")
                cache_stats["reasons"][f"debug_frame:{fallback_reason}"] = int(
                    cache_stats["reasons"].get(f"debug_frame:{fallback_reason}", 0)
                ) + 1
                if fallback_cache_stats.get("used"):
                    cache_stats["used"] += 1
                output = fallback_row
                output["vlm_video_error"] = str(video_exc)
                caption, labels, numeric, response = _run_vlm_generation(
                    output,
                    input_mode="debug_frame",
                    fps=fps,
                    max_pixels=max_pixels,
                    processor=processor,
                    model=model,
                    process_vision_info=process_vision_info,
                )
                used_input_mode = "debug_frame_fallback"
            output["vlm_model"] = model_id
            output["vlm_input_mode"] = used_input_mode
            output["vlm_caption"] = caption
            output["vlm_labels"] = labels
            output["numeric_features"] = numeric
            output["vlm_raw_response"] = response
            output["feature_status"] = "vlm_complete"
            processed += 1
        except Exception as exc:  # pragma: no cover - protects long Colab runs.
            output["feature_status"] = "vlm_failed"
            output["vlm_error"] = str(exc)
            failures.append({"clip_id": output.get("clip_id"), "error": str(exc)})
        output.pop("_runtime_clip_path", None)
        output.pop("_runtime_debug_frame_path", None)
        updated_rows.append(output)
        _write_vlm_feature_rows(manifest, updated_rows + rows[len(updated_rows) :])
        write_json(
            {
                "schema_version": "hf_vlm_captioning_progress_v1",
                "vlm_feature_id": vlm_feature_id,
                "model_id": model_id,
                "input_mode": input_mode,
                "video_reader_backend": video_reader_backend,
                "fallback_to_debug_frame": fallback_to_debug_frame,
                "processed": processed,
                "target_complete_rows": max_rows,
                "complete_rows_so_far": initial_complete_rows + processed,
                "rows": len(rows),
                "cache_stats": cache_stats,
                "failures": failures[:50],
            },
            summary,
        )

    _write_vlm_feature_rows(manifest, updated_rows)
    write_json(
        {
            "schema_version": "hf_vlm_captioning_summary_v1",
            "vlm_feature_id": vlm_feature_id,
            "model_id": model_id,
            "input_mode": input_mode,
            "processed": processed,
            "initial_complete_rows": initial_complete_rows,
            "target_complete_rows": max_rows,
            "video_reader_backend": video_reader_backend,
            "fallback_to_debug_frame": fallback_to_debug_frame,
            "rows": len(rows),
            "complete_rows": sum(1 for row in updated_rows if row.get("feature_status") == "vlm_complete"),
            "failed_rows": sum(1 for row in updated_rows if row.get("feature_status") == "vlm_failed"),
            "cache_dir": None if cache_dir is None else str(cache_dir),
            "cache_inputs": cache_inputs,
            "cache_min_free_disk_gb": cache_min_free_disk_gb,
            "cache_max_file_mb": cache_max_file_mb,
            "cache_stats": cache_stats,
            "failures": failures[:100],
            "outputs": {"manifest": str(manifest), "summary": str(summary)},
        },
        summary,
    )
    return {"manifest": manifest, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fill VLM mechanics manifest with HF VLM outputs.")
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/baseball_vision")
    parser.add_argument("--vlm-feature-id", default="vlm_mechanics_v1")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--input-mode", choices=("clip_video", "debug_frame"), default="clip_video")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-pixels", type=int, default=360 * 420)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--video-reader-backend", default=None)
    parser.add_argument("--no-debug-frame-fallback", action="store_true")
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--output-format", choices=("parquet", "jsonl", "json", "csv"), default="parquet")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--cache-inputs", action="store_true")
    parser.add_argument("--cache-min-free-disk-gb", type=float, default=20.0)
    parser.add_argument("--cache-max-file-mb", type=float, default=None)
    args = parser.parse_args(argv)
    outputs = fill_vlm_manifest_with_hf(
        args.base_dir,
        vlm_feature_id=args.vlm_feature_id,
        model_id=args.model_id,
        input_mode=args.input_mode,
        max_rows=args.max_rows,
        fps=args.fps,
        max_pixels=args.max_pixels,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        video_reader_backend=args.video_reader_backend,
        fallback_to_debug_frame=not args.no_debug_frame_fallback,
        token_env=args.token_env,
        output_suffix="." + args.output_format,
        cache_dir=args.cache_dir,
        cache_inputs=args.cache_inputs,
        cache_min_free_disk_gb=args.cache_min_free_disk_gb,
        cache_max_file_mb=args.cache_max_file_mb,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
