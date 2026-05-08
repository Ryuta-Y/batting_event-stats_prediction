"""Create the standard Drive/cache directory tree for Colab runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from sport_pipeline.artifact_check import write_json
from sport_pipeline.colab_paths import (
    ARTIFACT_DIRECTORIES,
    BASE_DIR_DEFAULT,
    CACHE_DIR_DEFAULT,
    REPO_DIR_DEFAULT,
    ensure_artifact_directories,
)


def initialize_drive_tree(
    base_dir: str | Path = BASE_DIR_DEFAULT,
    cache_dir: str | Path = CACHE_DIR_DEFAULT,
    repo_dir: str | Path = REPO_DIR_DEFAULT,
    output_json: str | Path | None = None,
) -> dict:
    """Create expected artifact directories and write an init report."""

    created = ensure_artifact_directories(base_dir=base_dir, cache_dir=cache_dir)
    output = Path(output_json) if output_json else Path(base_dir) / "reports/preflight/init_drive.json"
    payload = {
        "schema_version": "init_drive_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_dir": str(Path(repo_dir)),
        "base_dir": str(Path(base_dir)),
        "cache_dir": str(Path(cache_dir)),
        "artifact_directories": list(ARTIFACT_DIRECTORIES),
        "created_or_existing_paths": [str(path) for path in created],
        "next_notebook": "notebooks/30_cpu_data_sources_labels_reuse.ipynb",
    }
    write_json(payload, output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize Drive/cache artifact directories.")
    parser.add_argument("--base-dir", default=str(BASE_DIR_DEFAULT))
    parser.add_argument("--cache-dir", default=str(CACHE_DIR_DEFAULT))
    parser.add_argument("--repo-dir", default=str(REPO_DIR_DEFAULT))
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    result = initialize_drive_tree(args.base_dir, args.cache_dir, args.repo_dir, args.output_json)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
