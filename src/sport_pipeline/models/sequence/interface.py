"""Dependency-free sequence model interfaces.

The real TCN training implementation belongs in Colab. These dataclasses make
the expected inputs, heads, and aggregation metadata explicit for downstream
agents without pulling in torch locally.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TCNBaselineConfig:
    model_family: str = "tcn"
    input_artifact: str = "features/structured_sequence_v1/manifest.parquet"
    hidden_channels: tuple[int, ...] = (64, 128, 128)
    kernel_size: int = 5
    dropout: float = 0.20
    pooling: str = "attention_pooling"
    use_phase_auxiliary: bool = True
    use_contact_auxiliary: bool = True
    target_registry_path: str = "configs/targets/target_registry_v1.yaml"


@dataclass(frozen=True)
class SequenceModelInterface:
    config: TCNBaselineConfig = TCNBaselineConfig()

    @property
    def prediction_level(self) -> str:
        return "event"

    @property
    def supported_heads(self) -> tuple[str, ...]:
        return ("ev", "la", "hard_hit", "barrel", "xba", "xwoba")

    @property
    def aggregate_heads(self) -> tuple[str, ...]:
        return ("ops", "obp", "slg", "avg_ev", "hard_hit_rate", "barrel_rate")

