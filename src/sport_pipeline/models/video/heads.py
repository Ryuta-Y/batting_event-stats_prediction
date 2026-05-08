"""Target-registry driven lightweight heads for image/video baselines.

These helpers are dependency-free. Real frozen encoder inference should run in
Colab; local tests only verify target handling, shapes, masks, and
predictions_v1 mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

from sport_pipeline.evaluation.target_registry import TargetSpec


EVENT_VIDEO_TARGETS = ("ev", "la", "hard_hit", "barrel", "xba", "xwoba")


@dataclass(frozen=True)
class BaselineHeadSpec:
    name: str
    kind: str
    loss: str
    column: str
    required: bool
    requires_pa_manifest: bool = False


def build_event_head_specs(target_registry: Mapping[str, TargetSpec]) -> tuple[BaselineHeadSpec, ...]:
    """Build event-level head specs and deliberately exclude OPS."""

    specs: list[BaselineHeadSpec] = []
    for name in EVENT_VIDEO_TARGETS:
        if name not in target_registry:
            continue
        target = target_registry[name]
        if target.level != "event":
            continue
        specs.append(
            BaselineHeadSpec(
                name=target.name,
                kind=target.kind,
                loss=target.loss,
                column=target.column,
                required=target.required,
                requires_pa_manifest=target.requires_pa_manifest,
            )
        )
    return tuple(specs)


def validate_no_event_ops(head_specs: Iterable[BaselineHeadSpec]) -> None:
    """Reject accidental event-level OPS construction."""

    for spec in head_specs:
        if spec.name == "ops":
            raise ValueError("OPS must not be an event-level image/video head")


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _name_seed(name: str) -> float:
    return (sum(ord(char) for char in name) % 17 + 3) / 100.0


class LightweightHead:
    """Deterministic MLP-head stand-in for local shape tests."""

    def __init__(self, spec: BaselineHeadSpec, input_dim: int) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.spec = spec
        self.input_dim = input_dim
        seed = _name_seed(spec.name)
        self.weights = [seed * (index + 1) / input_dim for index in range(input_dim)]
        self.bias = seed

    def predict(self, features: list[list[float]]) -> list[float]:
        """Return one scalar prediction per feature row."""

        outputs: list[float] = []
        for row in features:
            if len(row) != self.input_dim:
                raise ValueError("feature dimension mismatch")
            score = sum(float(value) * weight for value, weight in zip(row, self.weights)) + self.bias
            if self.spec.kind in {"binary", "probability"}:
                outputs.append(_sigmoid(score))
            else:
                outputs.append(score)
        return outputs


class MultiTargetLightweightHead:
    """Registry-driven collection of event heads."""

    def __init__(self, head_specs: Iterable[BaselineHeadSpec], input_dim: int) -> None:
        specs = tuple(head_specs)
        validate_no_event_ops(specs)
        self.head_specs = specs
        self.input_dim = input_dim
        self.heads = {spec.name: LightweightHead(spec, input_dim=input_dim) for spec in specs}

    @property
    def target_names(self) -> tuple[str, ...]:
        return tuple(self.heads.keys())

    def predict(self, features: list[list[float]]) -> dict[str, list[float]]:
        return {name: head.predict(features) for name, head in self.heads.items()}


def build_loss_masks(
    samples: Iterable[dict],
    head_specs: Iterable[BaselineHeadSpec],
) -> dict[str, list[bool]]:
    """Build per-head loss masks, keeping optional missing labels out of loss."""

    sample_rows = list(samples)
    masks: dict[str, list[bool]] = {}
    for spec in head_specs:
        values: list[bool] = []
        availability_key = f"target_{spec.name}_available"
        for sample in sample_rows:
            if availability_key in sample:
                values.append(bool(sample[availability_key]))
            elif spec.required:
                values.append(sample.get(spec.column) is not None)
            else:
                values.append(sample.get(spec.column) is not None)
        masks[spec.name] = values
    return masks

