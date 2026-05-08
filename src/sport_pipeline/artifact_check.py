"""Artifact existence checks used by Colab notebooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from sport_pipeline.colab_paths import BASE_DIR_DEFAULT, expected_artifacts_for_stage


def check_artifacts(
    base_dir: str | Path = BASE_DIR_DEFAULT,
    artifacts: Iterable[str] = (),
) -> dict:
    """Return existence status for artifacts under the Drive artifact root."""

    root = Path(base_dir)
    entries = []
    for relative in artifacts:
        path = root / relative
        entries.append(
            {
                "artifact": relative,
                "path": str(path),
                "exists": path.exists(),
                "is_file": path.is_file(),
                "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
                "missing_hint_ja": None if path.exists() else missing_hint(relative),
            }
        )
    return {
        "base_dir": str(root),
        "all_present": all(entry["exists"] for entry in entries),
        "artifacts": entries,
    }


def check_stage_artifacts(
    stage: str,
    base_dir: str | Path = BASE_DIR_DEFAULT,
    run_id: str | None = None,
) -> dict:
    """Check artifacts for a configured stage."""

    return check_artifacts(base_dir=base_dir, artifacts=expected_artifacts_for_stage(stage, run_id=run_id))


def missing_hint(relative_path: str) -> str:
    """Return a short Japanese hint for a missing artifact."""

    if relative_path.endswith("check_env.json"):
        return "00_check_env.ipynb または python -m sport_pipeline.check_env を実行してください。"
    if relative_path.endswith("init_drive.json"):
        return "01_init_drive.ipynb または python -m sport_pipeline.init_drive を実行してください。"
    if "bbe_events_v1.parquet" in relative_path:
        return "11_download_statcast_and_video_sources.ipynb で Statcast BBE manifest を作成してください。"
    if "player_group_split_v1.parquet" in relative_path or "temporal_split_v1.parquet" in relative_path:
        return "split builder を実行し、leakage を避けた split artifact を作成してください。"
    if "video_sources_v1.parquet" in relative_path:
        return "11_download_statcast_and_video_sources.ipynb で video candidate manifest を作成してください。"
    if "clips_v1.parquet" in relative_path or "candidate_segments_v1.parquet" in relative_path:
        return "12_full_cv_preprocess.ipynb で downloaded/local video から clip artifact を作成してください。"
    if (
        "structured_sequence_v1" in relative_path
        or "structured_sequence_" in relative_path
        or "event_with_player_prior_v1" in relative_path
        or "event_with_player_prior_" in relative_path
        or "clip_embedding_v1" in relative_path
        or "clip_embedding_" in relative_path
        or "player_season_embedding_v1" in relative_path
        or "player_season_embedding_" in relative_path
        or "sequence_dataset_v1" in relative_path
        or "sequence_dataset_" in relative_path
    ):
        return "13_full_sequence_baseline.ipynb で sequence / prior artifact を作成してください。"
    if "features/video_embedding_v1" in relative_path or "features/video_embedding_" in relative_path or "features/video_lightweight" in relative_path:
        return "14_full_video_baseline.ipynb で video embedding artifact を作成してください。"
    if "cv_overlay_videos_v1" in relative_path or "cv_overlay_videos_" in relative_path:
        return "21_cv_overlay_videos.ipynb で YOLO/pose overlay mp4 を作成してください。"
    if "fusion_input_audit_v1" in relative_path:
        return "15_full_fusion.ipynb で fusion audit artifact を作成してください。"
    if "predictions_v1.parquet" in relative_path or "metrics_v1.json" in relative_path:
        if "context_catboost" in relative_path:
            return "05b_context_catboost_baseline.ipynb で CatBoost context baseline を作成してください。"
        return "05b_context_catboost_baseline.ipynb、13_full_sequence_baseline.ipynb、14_full_video_baseline.ipynb、15_full_fusion.ipynb のいずれかで predictions / metrics を作成してください。"
    if relative_path.endswith("index.html"):
        if "ablation_compare" in relative_path:
            return "20_video_ablation_compare.ipynb で video ablation report を作成してください。"
        return "09_report_builder.ipynb で HTML report を作成してください。"
    return "前段 notebook の出力 path と artifact 名を確認してください。"


def write_json(payload: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check pipeline artifacts under BASE_DIR.")
    parser.add_argument("--base-dir", default=str(BASE_DIR_DEFAULT))
    parser.add_argument("--stage", default="preflight")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    result = check_stage_artifacts(args.stage, base_dir=args.base_dir, run_id=args.run_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        write_json(result, args.output_json)


if __name__ == "__main__":
    main()
