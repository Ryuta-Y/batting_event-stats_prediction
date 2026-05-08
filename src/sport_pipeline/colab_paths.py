"""Shared Colab path and artifact utilities.

The notebooks stay thin by importing these helpers instead of duplicating path
setup and artifact checks in every notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


REPO_DIR_DEFAULT = Path("/content/drive/MyDrive/codex/batting_codex_handoff")
BASE_DIR_DEFAULT = Path("/content/drive/MyDrive/baseball_vision")
CACHE_DIR_DEFAULT = Path("/content/cache/baseball_vision")

ARTIFACT_DIRECTORIES = (
    "manifests",
    "manifests/splits",
    "raw_statcast",
    "raw_player_stats",
    "raw_videos",
    "annotations",
    "annotations/yolo",
    "annotations/yolo/bat_plate",
    "clips",
    "detections",
    "tracks",
    "pose2d",
    "objects",
    "homography",
    "features",
    "datasets",
    "datasets/player_season_targets",
    "shards",
    "models",
    "predictions",
    "reports",
    "reports/preflight",
    "reports/data_coverage",
    "reports/target_availability",
    "reports/clip_quality",
    "reports/experiment_compare",
    "reports/failure_browser",
    "reports/pipeline_dashboard",
    "reports/ablation_compare",
    "reports/research_outputs",
    "reports/method_evaluation",
    "reports/pipeline_io_map",
    "debug",
    "logs",
    "mlruns",
    "exports",
)

EXPECTED_ARTIFACTS = {
    "preflight": (
        "reports/preflight/check_env.json",
        "reports/preflight/init_drive.json",
    ),
    "data": (
        "manifests/bbe_events_v1.parquet",
        "manifests/splits/player_group_split_v1.parquet",
        "manifests/splits/temporal_split_v1.parquet",
    ),
    "video_probe": ("manifests/video_sources_v1.parquet",),
    "player_season_batting_stats": (
        "manifests/player_season_batting_v1.parquet",
        "reports/preflight/player_season_batting_stats_v1.json",
    ),
    "context_baseline": (
        "datasets/context_dataset_v1/manifest.parquet",
        "predictions/{run_id}/predictions_v1.parquet",
        "predictions/{run_id}/metrics_v1.json",
    ),
    "reports": (
        "reports/target_availability/{run_id}/index.html",
        "reports/experiment_compare/{run_id}/index.html",
        "reports/failure_browser/{run_id}/index.html",
        "reports/clip_quality/{run_id}/index.html",
        "reports/pipeline_dashboard/{run_id}/index.html",
    ),
    "full_cv": (
        "clips/{run_id}/candidate_segments_v1.parquet",
        "clips/{run_id}/clips_v1.parquet",
        "debug/{run_id}/debug_overlays_v1.parquet",
        "reports/preflight/full_cv_preprocess_{run_id}.json",
        "reports/preflight/full_cv_preprocess_{run_id}_progress.json",
    ),
    "sequence_baseline": (
        "features/{structured_sequence_feature_id}/manifest.parquet",
        "features/{structured_sequence_feature_id}/frames.parquet",
        "features/{clip_embedding_feature_id}/manifest.parquet",
        "features/{player_season_embedding_feature_id}/manifest.parquet",
        "datasets/{sequence_dataset_id}/manifest.parquet",
        "datasets/{event_with_prior_dataset_id}/manifest.parquet",
        "predictions/{run_id}/predictions_v1.parquet",
        "predictions/{run_id}/metrics_v1.json",
    ),
    "video_baseline": (
        "features/{video_lightweight_feature_id}/manifest.parquet",
        "predictions/{run_id}/predictions_v1.parquet",
        "predictions/{run_id}/metrics_v1.json",
    ),
    "raw_video_finetune": (
        "predictions/{run_id}/predictions_v1.parquet",
        "predictions/{run_id}/metrics_v1.json",
        "reports/preflight/raw_video_finetune_{run_id}.json",
        "reports/preflight/raw_video_finetune_{run_id}_progress.json",
        "models/video/{run_id}/checkpoint.pt",
    ),
    "fusion": (
        "predictions/{run_id}/predictions_v1.parquet",
        "predictions/{run_id}/metrics_v1.json",
        "predictions/{run_id}/fusion_input_audit_v1.parquet",
    ),
    "player_season_aggregate": (
        "datasets/player_season_targets/{run_id}/manifest.parquet",
        "predictions/{run_id}/predictions_v1.parquet",
        "predictions/{run_id}/metrics_v1.json",
        "reports/preflight/player_season_aggregate_{run_id}.json",
    ),
    "video_ablation": (
        "reports/ablation_compare/{run_id}/index.html",
        "reports/ablation_compare/{run_id}/summary.json",
    ),
    "cv_overlay": (
        "debug/{run_id}/cv_overlay_videos_v1.parquet",
        "debug/{run_id}/debug_overlays_v1.parquet",
        "reports/preflight/cv_overlay_videos_{run_id}.json",
        "reports/preflight/cv_overlay_videos_{run_id}_progress.json",
    ),
    "research_outputs": (
        "reports/research_outputs/{run_id}/index.html",
        "reports/research_outputs/{run_id}/summary.json",
        "reports/research_outputs/{run_id}/figures/method_overview.svg",
        "reports/research_outputs/{run_id}/tables/metrics_by_run_target.csv",
    ),
    "method_evaluation": (
        "reports/method_evaluation/{run_id}/index.html",
        "reports/method_evaluation/{run_id}/summary.json",
        "reports/method_evaluation/{run_id}/tables/method_metrics.csv",
        "reports/method_evaluation/{run_id}/tables/sample_counts.csv",
    ),
    "vlm_mechanics": (
        "features/{vlm_feature_id}/manifest.parquet",
        "reports/preflight/vlm_feature_template_{vlm_feature_id}.json",
        "reports/preflight/hf_vlm_captioning_{vlm_feature_id}.json",
        "predictions/{run_id}/predictions_v1.parquet",
        "predictions/{run_id}/metrics_v1.json",
    ),
}

DEFAULT_EXPECTED_ARTIFACT_IDS = {
    "structured_sequence_feature_id": "structured_sequence_mlb_2024_2026_v2",
    "clip_embedding_feature_id": "clip_embedding_mlb_2024_2026_v2",
    "player_season_embedding_feature_id": "player_season_embedding_mlb_2024_2026_v2",
    "sequence_dataset_id": "sequence_dataset_mlb_2024_2026_v2",
    "event_with_prior_dataset_id": "event_with_player_prior_mlb_2024_2026_v2",
    "video_lightweight_feature_id": "video_lightweight_features_mlb_2024_2026_v2",
    "video_embedding_feature_id": "video_embedding_mlb_2024_2026_v2",
    "image_embedding_feature_id": "image_embedding_mlb_2024_2026_v2",
    "vlm_feature_id": "vlm_mechanics_mlb_2024_2026_v2",
}


@dataclass(frozen=True)
class ColabPaths:
    repo_dir: Path = REPO_DIR_DEFAULT
    base_dir: Path = BASE_DIR_DEFAULT
    cache_dir: Path = CACHE_DIR_DEFAULT

    @classmethod
    def from_values(
        cls,
        repo_dir: str | Path | None = None,
        base_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ) -> "ColabPaths":
        return cls(
            repo_dir=Path(repo_dir) if repo_dir is not None else REPO_DIR_DEFAULT,
            base_dir=Path(base_dir) if base_dir is not None else BASE_DIR_DEFAULT,
            cache_dir=Path(cache_dir) if cache_dir is not None else CACHE_DIR_DEFAULT,
        )


def ensure_artifact_directories(
    base_dir: str | Path = BASE_DIR_DEFAULT,
    cache_dir: str | Path = CACHE_DIR_DEFAULT,
    directories: Iterable[str] = ARTIFACT_DIRECTORIES,
) -> list[Path]:
    """Create the standard Drive and cache artifact directories."""

    created: list[Path] = []
    roots = (Path(base_dir), Path(cache_dir))
    for root in roots:
        for relative in directories:
            path = root / relative
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)
    return created


def expected_artifacts_for_stage(
    stage: str,
    run_id: str | None = None,
    *,
    artifact_ids: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return expected artifact paths for a named pipeline stage."""

    if stage not in EXPECTED_ARTIFACTS:
        raise KeyError(f"Unknown artifact stage: {stage}")
    format_values = dict(DEFAULT_EXPECTED_ARTIFACT_IDS)
    if artifact_ids:
        format_values.update({str(key): str(value) for key, value in artifact_ids.items()})
    if run_id is not None:
        format_values["run_id"] = run_id
    resolved = []
    for template in EXPECTED_ARTIFACTS[stage]:
        try:
            resolved.append(template.format(**format_values))
        except KeyError:
            resolved.append(template)
    return tuple(resolved)
