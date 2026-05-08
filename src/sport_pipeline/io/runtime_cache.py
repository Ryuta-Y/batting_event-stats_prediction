"""Runtime-local file caching helpers for Colab video stages."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import shutil
from pathlib import Path


@dataclass(frozen=True)
class CacheFileResult:
    """Result of staging one file into the runtime cache."""

    source_path: Path
    path: Path
    used_cache: bool
    reason: str
    size_bytes: int


def safe_cache_key(value: str) -> str:
    """Return a stable filesystem-safe key for cache filenames."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned or "file"


def _cache_destination(cache_dir: Path, namespace: str, source_path: Path, key: str | None) -> Path:
    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:12]
    stem = safe_cache_key(key or source_path.stem)
    suffix = source_path.suffix or ".bin"
    return cache_dir / namespace / f"{stem}_{digest}{suffix}"


def cache_file(
    source_path: str | Path,
    *,
    cache_dir: str | Path | None,
    namespace: str,
    key: str | None = None,
    enabled: bool = True,
    max_file_bytes: int | None = None,
    min_free_disk_bytes: int = 20 * 1024**3,
) -> CacheFileResult:
    """Copy a file into local runtime cache when policy and disk space allow it."""

    source = Path(source_path)
    try:
        size = source.stat().st_size
    except OSError:
        return CacheFileResult(source, source, False, "source_missing_or_unstatable", 0)
    if not enabled or cache_dir is None:
        return CacheFileResult(source, source, False, "cache_disabled", size)
    if max_file_bytes is not None and size > max_file_bytes:
        return CacheFileResult(source, source, False, "source_too_large_for_cache", size)

    root = Path(cache_dir)
    destination = _cache_destination(root, namespace, source, key)
    try:
        usage = shutil.disk_usage(root if root.exists() else root.parent)
        if usage.free - size < min_free_disk_bytes:
            return CacheFileResult(source, source, False, "insufficient_cache_free_space", size)
    except OSError:
        return CacheFileResult(source, source, False, "cache_disk_unavailable", size)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == size and size > 0:
        return CacheFileResult(source, destination, True, "cache_hit", size)
    temp_path = destination.with_name(destination.name + ".tmp")
    try:
        shutil.copy2(source, temp_path)
        temp_path.replace(destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return CacheFileResult(source, destination, True, "cache_copied", size)


def cache_output_path(cache_dir: str | Path | None, *, namespace: str, filename: str, enabled: bool = True) -> Path | None:
    """Return a local output-cache path for a final artifact filename."""

    if not enabled or cache_dir is None:
        return None
    output = Path(cache_dir) / namespace / safe_cache_key(filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def publish_cached_file(cached_path: str | Path, final_path: str | Path) -> None:
    """Copy a runtime-cache output to its persistent final path."""

    cached = Path(cached_path)
    final = Path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, final)
