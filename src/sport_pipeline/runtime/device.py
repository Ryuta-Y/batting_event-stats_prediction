"""Small, optional torch-aware device helpers.

The helpers avoid importing torch at module import time so local tests stay
lightweight. Colab notebooks can use the returned `selected_device` to decide
whether a stage should run on CUDA or stop before heavy work starts.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeDeviceInfo:
    """Normalized runtime device metadata for notebooks and reports."""

    torch_available: bool
    cuda_available: bool
    selected_device: str
    require_gpu: bool
    cuda_visible_devices: str | None
    torch_version: str | None = None
    cuda_version: str | None = None
    cuda_device_count: int = 0
    current_device_index: int | None = None
    gpu_name: str | None = None
    gpu_total_memory_gb: float | None = None
    warning_ja: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable metadata."""

        return asdict(self)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def detect_runtime_device(prefer_gpu: bool = True, require_gpu: bool = False) -> RuntimeDeviceInfo:
    """Detect torch/CUDA state and select `cuda` or `cpu`.

    `require_gpu=True` does not raise. It records a warning so notebook cells can
    display a readable stop condition before any model download or training.
    """

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not _module_available("torch"):
        selected = "cpu"
        warning = "torch が import できないため CPU 実行になります。GPU stage の前に Colab runtime / package を確認してください。"
        if require_gpu:
            warning = "GPU 必須 stage ですが torch が import できません。Colab の GPU runtime と package を確認してください。"
        return RuntimeDeviceInfo(
            torch_available=False,
            cuda_available=False,
            selected_device=selected,
            require_gpu=require_gpu,
            cuda_visible_devices=cuda_visible_devices,
            warning_ja=warning,
        )

    try:
        import torch  # type: ignore

        cuda_available = bool(torch.cuda.is_available())
        selected = "cuda" if prefer_gpu and cuda_available else "cpu"
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
        current_index = int(torch.cuda.current_device()) if cuda_available else None
        props = torch.cuda.get_device_properties(current_index) if current_index is not None else None
        warning = None
        if require_gpu and selected != "cuda":
            warning = "GPU 必須 stage ですが CUDA が使えません。Colab の Runtime type を GPU に変更してください。"
        return RuntimeDeviceInfo(
            torch_available=True,
            torch_version=getattr(torch, "__version__", None),
            cuda_available=cuda_available,
            cuda_version=getattr(torch.version, "cuda", None),
            cuda_device_count=device_count,
            current_device_index=current_index,
            gpu_name=torch.cuda.get_device_name(current_index) if current_index is not None else None,
            gpu_total_memory_gb=round(float(props.total_memory) / (1024**3), 2) if props is not None else None,
            selected_device=selected,
            require_gpu=require_gpu,
            cuda_visible_devices=cuda_visible_devices,
            warning_ja=warning,
        )
    except Exception as exc:  # pragma: no cover - defensive for unusual runtimes
        return RuntimeDeviceInfo(
            torch_available=True,
            cuda_available=False,
            selected_device="cpu",
            require_gpu=require_gpu,
            cuda_visible_devices=cuda_visible_devices,
            warning_ja=f"torch はありますが CUDA 確認に失敗しました: {exc}",
        )


def summarize_runtime_device(prefer_gpu: bool = True, require_gpu: bool = False) -> dict[str, Any]:
    """Return runtime device metadata as a dict."""

    return detect_runtime_device(prefer_gpu=prefer_gpu, require_gpu=require_gpu).to_dict()
