"""Run isolation checks for Colab/Drive research artifacts.

The project stores large outputs in Drive. Most outputs are scoped by run id,
but feature and dataset folders used by multiple notebooks need their own
namespace ids when moving from v1 to v2 or when growing the clip tranche.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

from sport_pipeline.artifact_check import write_json
from sport_pipeline.pipeline.run_profile import DEFAULT_ARTIFACT_NAMESPACE, artifact_namespace, load_run_profile


RUN_ID_KEYS = (
    "context_run_id",
    "recommended_context_run_id",
    "full_run_id",
    "object_detector_run_id",
    "sequence_run_id",
    "sequence_tcn_run_id",
    "video_lightweight_run_id",
    "video_frozen_run_id",
    "video_finetune_run_id",
    "video_run_id",
    "video_ablation_report_id",
    "method_evaluation_report_id",
    "player_season_run_id",
    "vlm_run_id",
    "fusion_run_id",
)

NAMESPACE_PATHS = {
    "structured_sequence_feature_id": "features/{value}/manifest.parquet",
    "clip_embedding_feature_id": "features/{value}/manifest.parquet",
    "player_season_embedding_feature_id": "features/{value}/manifest.parquet",
    "sequence_dataset_id": "datasets/{value}/manifest.parquet",
    "event_with_prior_dataset_id": "datasets/{value}/manifest.parquet",
    "video_lightweight_feature_id": "features/{value}/manifest.parquet",
    "video_embedding_feature_id": "features/{value}/manifest.parquet",
    "image_embedding_feature_id": "features/{value}/manifest.parquet",
    "vlm_feature_id": "features/{value}/manifest.parquet",
}

NAMESPACE_CONSUMERS = {
    "structured_sequence_feature_id": "13/17 write it; 18 reads it",
    "clip_embedding_feature_id": "13 writes player-event clip embeddings",
    "player_season_embedding_feature_id": "13 writes player-season mechanics priors",
    "sequence_dataset_id": "13 writes event sequence dataset rows",
    "event_with_prior_dataset_id": "13 writes event rows with prior; 18 reads it",
    "video_lightweight_feature_id": "14 writes OpenCV lightweight video feature rows",
    "video_embedding_feature_id": "19 writes/reuses VideoMAE embeddings",
    "image_embedding_feature_id": "19 writes/reuses DINO contact-frame embeddings",
    "vlm_feature_id": "24 writes VLM prompt rows and reads filled VLM caption/tag features",
}


def _version_token(value: str) -> str | None:
    match = re.search(r"_(v\d+)$", value)
    return match.group(1) if match else None


def _replace_version(value: str, version: str) -> str:
    if re.search(r"_v\d+$", value):
        return re.sub(r"_v\d+$", f"_{version}", value)
    return f"{value}_{version}"


def _full_run_stem(full_run_id: str, version: str) -> str:
    stem = re.sub(r"_full_v\d+$", "", full_run_id)
    stem = re.sub(r"_v\d+$", "", stem)
    return f"{stem}_{version}"


def make_next_run_profile(
    profile: dict[str, Any],
    *,
    version: str = "v2",
    max_files: int | None = None,
    max_clips: int | None = None,
) -> dict[str, Any]:
    """Return a copied profile with run ids and shared artifacts moved to versioned names."""

    next_profile = copy.deepcopy(profile)
    previous_namespace = artifact_namespace(profile)
    run_ids = dict(next_profile.get("run_ids", {}))
    original_run_ids = dict(run_ids)
    for key, value in list(run_ids.items()):
        if isinstance(value, str) and value:
            run_ids[key] = _replace_version(value, version)
    next_profile["run_ids"] = run_ids

    full_run_id = str(run_ids.get("full_run_id", f"mlb_2024_2026_full_{version}"))
    stem = _full_run_stem(full_run_id, version)
    next_namespace = {
        "structured_sequence_feature_id": f"structured_sequence_{stem}",
        "clip_embedding_feature_id": f"clip_embedding_{stem}",
        "player_season_embedding_feature_id": f"player_season_embedding_{stem}",
        "sequence_dataset_id": f"sequence_dataset_{stem}",
        "event_with_prior_dataset_id": f"event_with_player_prior_{stem}",
        "video_lightweight_feature_id": f"video_lightweight_features_{stem}",
        "video_embedding_feature_id": f"video_embedding_{stem}",
        "image_embedding_feature_id": f"image_embedding_{stem}",
        "vlm_feature_id": f"vlm_mechanics_{stem}",
    }
    next_profile["artifact_namespace"] = next_namespace

    execution = dict(next_profile.get("execution", {}))
    previous_full_run_id = str(original_run_ids.get("full_run_id") or "")
    if previous_full_run_id:
        video_reuse = dict(execution.get("video_reuse", {}))
        source_full_run_ids = [str(item) for item in video_reuse.get("source_full_run_ids", [])]
        if previous_full_run_id not in source_full_run_ids:
            source_full_run_ids.insert(0, previous_full_run_id)
        video_reuse["source_full_run_ids"] = source_full_run_ids
        video_reuse["reuse_previous_downloads"] = True
        video_reuse.setdefault("preserve_existing_canonical_manifest_on_merge", True)
        execution["video_reuse"] = video_reuse
    if max_files is not None:
        video_download = dict(execution.get("video_download", {}))
        video_download["max_files"] = int(max_files)
        execution["video_download"] = video_download
    if max_clips is not None:
        for stage in ("deep_cv", "frozen_visual_encoder", "raw_video_finetune", "vlm_mechanics"):
            settings = dict(execution.get(stage, {}))
            settings["max_clips"] = int(max_clips)
            execution[stage] = settings
    for stage in ("fusion",):
        settings = dict(execution.get(stage, {}))
        for source_key in ("source_runs", "fallback_source_runs"):
            source_runs = settings.get(source_key)
            if isinstance(source_runs, list):
                settings[source_key] = [_replace_version(str(item), version) for item in source_runs]
        execution[stage] = settings
    next_profile["execution"] = execution
    replacements = {
        **{str(old): str(run_ids.get(key, old)) for key, old in original_run_ids.items()},
        **{str(old): str(next_namespace.get(key, old)) for key, old in previous_namespace.items()},
    }
    artifact_groups = next_profile.get("artifact_groups")
    if isinstance(artifact_groups, list):
        rewritten_groups = []
        for group in artifact_groups:
            if not isinstance(group, dict):
                rewritten_groups.append(group)
                continue
            rewritten = dict(group)
            artifacts = rewritten.get("artifacts")
            if isinstance(artifacts, list):
                new_artifacts = []
                for artifact in artifacts:
                    updated = str(artifact)
                    for old, new in replacements.items():
                        updated = updated.replace(old, new)
                    new_artifacts.append(updated)
                rewritten["artifacts"] = new_artifacts
            rewritten_groups.append(rewritten)
        next_profile["artifact_groups"] = rewritten_groups
    next_profile["schema_version"] = f"{next_profile.get('schema_version', 'real_colab_run_profile')}_{version}_derived"
    return next_profile


def audit_run_isolation(
    profile: dict[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable v1/v2 mixing audit."""

    run_ids = {key: str(profile.get("run_ids", {}).get(key)) for key in RUN_ID_KEYS if profile.get("run_ids", {}).get(key)}
    versions = sorted({token for value in run_ids.values() if (token := _version_token(value))})
    namespace = artifact_namespace(profile)
    checks: list[dict[str, Any]] = []
    severity_rank = {"ok": 0, "warn": 1, "fail": 2}
    base = Path(base_dir) if base_dir is not None else None

    duplicate_values: dict[str, list[str]] = {}
    for key, value in run_ids.items():
        duplicate_values.setdefault(value, []).append(key)
    for value, keys in sorted(duplicate_values.items()):
        if len(keys) > 1 and set(keys) != {"context_run_id", "recommended_context_run_id"} and set(keys) != {"video_frozen_run_id", "video_run_id"}:
            checks.append(
                {
                    "status": "warn",
                    "kind": "duplicate_run_id",
                    "value": value,
                    "keys": keys,
                    "message_ja": "複数の run_id key が同じ値です。意図した alias でなければ v2 実行で上書き混線します。",
                }
            )

    expected_version = versions[-1] if versions else None
    for key, default_value in DEFAULT_ARTIFACT_NAMESPACE.items():
        value = namespace.get(key, default_value)
        path_template = NAMESPACE_PATHS[key]
        relative_path = path_template.format(value=value)
        exists = (base / relative_path).exists() if base is not None else None
        status = "ok"
        reason = "namespace is explicit"
        if expected_version and value == default_value and expected_version != "v1":
            status = "fail"
            reason = f"{key} still uses shared default {default_value} while run_ids look like {expected_version}"
        elif expected_version and not str(value).endswith(expected_version):
            status = "warn"
            reason = f"{key} does not end with {expected_version}; verify this is intentional"
        elif value == default_value:
            status = "warn"
            reason = f"{key} uses legacy shared default; safe for the current v1 rerun, risky for v2 unless changed"
        checks.append(
            {
                "status": status,
                "kind": "artifact_namespace",
                "key": key,
                "value": value,
                "default_value": default_value,
                "relative_path": relative_path,
                "exists": exists,
                "consumer": NAMESPACE_CONSUMERS[key],
                "message_ja": reason,
            }
        )

    if len(versions) > 1:
        checks.append(
            {
                "status": "warn",
                "kind": "mixed_run_id_versions",
                "versions": versions,
                "message_ja": "run_ids の suffix に複数 version が混在しています。意図した比較でなければ source_runs を確認してください。",
            }
        )

    if expected_version and expected_version != "v1":
        video_reuse = profile.get("execution", {}).get("video_reuse", {})
        source_full_run_ids = video_reuse.get("source_full_run_ids") if isinstance(video_reuse, dict) else None
        reuse_enabled = bool(isinstance(video_reuse, dict) and video_reuse.get("reuse_previous_downloads", False))
        checks.append(
            {
                "status": "ok" if reuse_enabled and source_full_run_ids else "warn",
                "kind": "video_reuse",
                "source_full_run_ids": source_full_run_ids or [],
                "message_ja": (
                    "v2 は previous-run download manifest から既存動画を再利用する設定です。"
                    if reuse_enabled and source_full_run_ids
                    else "v2 で v1 raw video を再利用する設定がありません。再ダウンロードや manifest 分離漏れを確認してください。"
                ),
            }
        )

    max_status = max((item["status"] for item in checks), key=lambda value: severity_rank[value], default="ok")
    return {
        "schema_version": "run_isolation_audit_v1",
        "overall_status": max_status,
        "run_id_versions": versions,
        "run_ids": run_ids,
        "artifact_namespace": namespace,
        "checks": checks,
        "recommendation_ja": (
            "v2 や 500本超への拡張では run_ids だけでなく artifact_namespace も v2 名にしてください。"
            "特に features/structured_sequence, player_season_embedding, video_embedding, datasets/event_with_player_prior は共有名のままだと混線します。"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit and prepare run-isolated Colab profiles.")
    parser.add_argument("--run-profile", default=None)
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--next-version", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--write-next-profile", default=None)
    args = parser.parse_args(argv)

    profile = load_run_profile(args.run_profile)
    if args.next_version:
        profile = make_next_run_profile(
            profile,
            version=args.next_version,
            max_files=args.max_files,
            max_clips=args.max_clips,
        )
        if args.write_next_profile:
            output_profile = Path(args.write_next_profile)
            output_profile.parent.mkdir(parents=True, exist_ok=True)
            output_profile.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    audit = audit_run_isolation(profile, base_dir=args.base_dir)
    if args.output:
        write_json(audit, Path(args.output))
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if audit["overall_status"] in {"ok", "warn"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
