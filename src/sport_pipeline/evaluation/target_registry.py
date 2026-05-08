"""Target registry loading for D1-owned evaluator contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TargetSpec:
    name: str
    group: str
    level: str
    column: str
    kind: str
    loss: str
    metrics: tuple[str, ...]
    required: bool
    requires_pa_manifest: bool = False


def _load_json_compatible_yaml(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ValueError(
                f"{path} is not JSON-compatible YAML and PyYAML is not installed"
            ) from exc
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} did not parse to a mapping")
        return loaded


def load_target_registry(path: str | Path) -> dict[str, TargetSpec]:
    """Load registry targets keyed by target name."""

    payload = _load_json_compatible_yaml(path)
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, dict):
        raise ValueError("target registry must contain a targets mapping")
    targets: dict[str, TargetSpec] = {}
    for name, raw in raw_targets.items():
        if not isinstance(raw, dict):
            raise ValueError(f"target {name} must be a mapping")
        targets[name] = TargetSpec(
            name=name,
            group=str(raw["group"]),
            level=str(raw["level"]),
            column=str(raw["column"]),
            kind=str(raw["kind"]),
            loss=str(raw["loss"]),
            metrics=tuple(str(metric) for metric in raw.get("metrics", [])),
            required=bool(raw.get("required", False)),
            requires_pa_manifest=bool(raw.get("requires_pa_manifest", False)),
        )
    return targets

