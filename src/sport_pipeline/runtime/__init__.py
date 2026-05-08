"""Runtime helpers for Colab device selection and preflight checks."""

from sport_pipeline.runtime.device import RuntimeDeviceInfo, detect_runtime_device, summarize_runtime_device

__all__ = ["RuntimeDeviceInfo", "detect_runtime_device", "summarize_runtime_device"]
