"""Colab preflight environment check."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from sport_pipeline.artifact_check import write_json
from sport_pipeline.colab_paths import BASE_DIR_DEFAULT, CACHE_DIR_DEFAULT, REPO_DIR_DEFAULT
from sport_pipeline.runtime import summarize_runtime_device


OPTIONAL_IMPORTS = (
    "sport_pipeline",
    "pandas",
    "numpy",
    "pyarrow",
    "sklearn",
    "catboost",
    "torch",
)


def in_colab() -> bool:
    """Return true when running inside Google Colab."""

    return _module_available("google.colab")


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def _gpu_info() -> dict:
    return summarize_runtime_device(prefer_gpu=True, require_gpu=False)


def run_check_env(
    base_dir: str | Path = BASE_DIR_DEFAULT,
    cache_dir: str | Path = CACHE_DIR_DEFAULT,
    repo_dir: str | Path = REPO_DIR_DEFAULT,
    output_json: str | Path | None = None,
) -> dict:
    """Collect local/Colab runtime status and write check_env.json."""

    base = Path(base_dir)
    cache = Path(cache_dir)
    repo = Path(repo_dir)
    base.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": "check_env_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "repo_dir": str(repo),
            "base_dir": str(base),
            "cache_dir": str(cache),
            "cwd": str(Path.cwd()),
            "repo_dir_exists": repo.exists(),
            "base_dir_exists": base.exists(),
            "cache_dir_exists": cache.exists(),
        },
        "runtime": {
            "in_colab": in_colab(),
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "imports": {name: _module_available(name) for name in OPTIONAL_IMPORTS},
        "gpu": _gpu_info(),
        "warnings_ja": [],
    }
    if repo.parts[-2:] == ("codex", "vision"):
        payload["warnings_ja"].append("旧 Colab code root を使っています。REPO_DIR_DEFAULT の固定値に修正してください。")
    if not repo.exists():
        payload["warnings_ja"].append("REPO_DIR が存在しません。Drive mount と code root を確認してください。")
    if not _module_available("sport_pipeline"):
        payload["warnings_ja"].append("sport_pipeline を import できません。sys.path に REPO_DIR/src を追加してください。")

    output = Path(output_json) if output_json else base / "reports/preflight/check_env.json"
    write_json(payload, output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Colab preflight checks.")
    parser.add_argument("--base-dir", default=str(BASE_DIR_DEFAULT))
    parser.add_argument("--cache-dir", default=str(CACHE_DIR_DEFAULT))
    parser.add_argument("--repo-dir", default=str(REPO_DIR_DEFAULT))
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    result = run_check_env(args.base_dir, args.cache_dir, args.repo_dir, args.output_json)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
