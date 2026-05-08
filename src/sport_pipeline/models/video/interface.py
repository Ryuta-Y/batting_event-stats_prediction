"""Frozen video encoder baseline interface.

The real VideoMAE encoder runs in Colab. This module keeps the local contract
testable without model downloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sport_pipeline.evaluation.target_registry import load_target_registry
from sport_pipeline.models.video.heads import MultiTargetLightweightHead, build_event_head_specs, build_loss_masks


@dataclass(frozen=True)
class FrozenVideoBaselineConfig:
    encoder_name: str = "videomae_base_kinetics_frozen"
    encoder_family: str = "VideoMAE"
    model_id: str = "MCG-NJU/videomae-base-finetuned-kinetics"
    input_num_frames: int = 16
    input_resolution: int = 224
    frame_sampling: str = "uniform_contact_centered"
    clip_source: str = "batter_crop_or_raw_clip"
    head_name: str = "lightweight_mlp"
    train_encoder: bool = False
    target_registry_path: str = "configs/targets/target_registry_v1.yaml"


class FrozenVideoBaseline:
    """Registry-driven frozen-video baseline interface."""

    def __init__(
        self,
        input_dim: int,
        target_registry_path: str | Path = "configs/targets/target_registry_v1.yaml",
        config: FrozenVideoBaselineConfig | None = None,
    ) -> None:
        self.config = config or FrozenVideoBaselineConfig(target_registry_path=str(target_registry_path))
        self.targets = load_target_registry(target_registry_path)
        self.head_specs = build_event_head_specs(self.targets)
        self.head = MultiTargetLightweightHead(self.head_specs, input_dim=input_dim)

    def predict_from_features(self, features: list[list[float]]) -> dict[str, list[float]]:
        """Predict event-level targets from frozen video features."""

        return self.head.predict(features)

    def loss_masks(self, samples: list[dict]) -> dict[str, list[bool]]:
        return build_loss_masks(samples, self.head_specs)
