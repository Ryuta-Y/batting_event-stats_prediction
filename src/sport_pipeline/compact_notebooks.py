"""Helpers for compact Colab wrapper notebooks.

The compact notebooks call the thin stage notebooks with `%run`. This helper
adds consistent console logging, JSONL progress logs, completed-stage skipping,
and progress snapshots from the stage-level preflight JSON files.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Iterable


ProgressPath = str | Path


DRIVE_TRANSPORT_HINT = (
    "Colab Drive mount appears to be disconnected (Errno 107: Transport endpoint is not connected). "
    "This is a Colab/Drive FUSE mount problem, not a model/output validation failure. "
    "Remount Drive or restart the runtime, then rerun the compact notebook; completed stages are resumable."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_transport_endpoint_error(exc: BaseException) -> bool:
    return getattr(exc, "errno", None) == 107 or "Transport endpoint is not connected" in str(exc)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _progress_pair(payload: dict[str, Any]) -> tuple[float, float, str] | None:
    for current_key, total_key, label in (
        ("completed_chunks", "total_chunks", "chunks"),
        ("completed_events", "total_events", "events"),
        ("overall_completed", "overall_total_planned", "videos"),
        ("seen_sources", "total_sources", "sources"),
        ("seen_clips", "selected_clips", "clips"),
        ("processed", "rows", "rows"),
        ("completed_epochs", "max_epochs", "epochs"),
        ("rendered_overlays", "selected_clips", "overlays"),
    ):
        current = payload.get(current_key)
        total = payload.get(total_key)
        if isinstance(current, (int, float)) and isinstance(total, (int, float)) and total:
            return float(current), float(total), label
    if payload.get("schema_version") == "deep_cv_progress_v1":
        total = payload.get("resolved_clip_files") or payload.get("input_clip_rows")
        if isinstance(total, (int, float)) and total:
            completed = max(
                len(payload.get("completed_yolo_clip_ids") or []),
                len(payload.get("completed_object_clip_ids") or []),
                len(payload.get("completed_pose_clip_ids") or []),
            )
            return float(completed), float(total), "clips"
    return None


def progress_snapshots(base_dir: str | Path, progress_files: Iterable[ProgressPath]) -> list[dict[str, Any]]:
    base = Path(base_dir)
    snapshots: list[dict[str, Any]] = []
    for item in progress_files:
        path = Path(item)
        if not path.is_absolute():
            path = base / path
        snapshot: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        payload = _read_json(path) if path.exists() else None
        if payload:
            snapshot["status"] = payload.get("status") or payload.get("overall_status") or payload.get("schema_version")
            pair = _progress_pair(payload)
            if pair is not None:
                current, total, unit = pair
                snapshot.update(
                    {
                        "current": int(current) if current.is_integer() else current,
                        "total": int(total) if total.is_integer() else total,
                        "percent": round(100.0 * current / total, 3) if total else None,
                        "unit": unit,
                    }
                )
            for key in (
                "raw_rows_so_far",
                "bbe_rows_so_far",
                "media_url_rows",
                "clean_trainable_clips",
                "prediction_rows",
                "player_season_samples",
                "metric_rows",
                "same_sample_metric_rows",
            ):
                if key in payload:
                    snapshot[key] = payload[key]
        snapshots.append(snapshot)
    return snapshots


def outputs_complete(base_dir: str | Path, expected_outputs: Iterable[ProgressPath]) -> bool:
    outputs = list(expected_outputs)
    if not outputs:
        return False
    base = Path(base_dir)
    for item in outputs:
        path = Path(item)
        if not path.is_absolute():
            path = base / path
        if not path.exists():
            return False
    return True


class CompactRunLogger:
    """Small JSONL logger used by notebook wrappers 30-35."""

    def __init__(self, base_dir: str | Path, run_name: str, *, run_profile_name: str | None = None) -> None:
        self.base_dir = Path(base_dir)
        self.run_name = run_name
        self.run_profile_name = run_profile_name
        self.log_path = self.base_dir / "reports/preflight/compact_runs" / f"{run_name}.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, Any]) -> None:
        row = {
            "timestamp_utc": _now(),
            "compact_run": self.run_name,
            "run_profile": self.run_profile_name,
            **payload,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def print_snapshots(self, progress_files: Iterable[ProgressPath]) -> list[dict[str, Any]]:
        snapshots = progress_snapshots(self.base_dir, progress_files)
        for snapshot in snapshots:
            if not snapshot["exists"]:
                print("progress missing:", snapshot["path"])
                continue
            if "current" in snapshot and "total" in snapshot:
                print(
                    "progress:",
                    snapshot["path"],
                    f"{snapshot['current']}/{snapshot['total']}",
                    f"{snapshot.get('percent', 0):.1f}%",
                    snapshot.get("unit", ""),
                    "status=" + str(snapshot.get("status")),
                )
            else:
                print("progress:", snapshot["path"], "status=" + str(snapshot.get("status")))
        return snapshots

    def run_stage(
        self,
        *,
        ipython: Any,
        notebook_dir: str | Path,
        index: int,
        total: int,
        notebook: str,
        label: str | None = None,
        enabled: bool = True,
        force: bool = False,
        expected_outputs: Iterable[ProgressPath] = (),
        progress_files: Iterable[ProgressPath] = (),
    ) -> None:
        stage_label = label or notebook
        stage_percent = 100.0 * (index - 1) / total if total else 0.0
        stage_payload = {
            "stage_index": index,
            "stage_total": total,
            "stage_percent_before": round(stage_percent, 3),
            "notebook": notebook,
            "label": stage_label,
            "expected_outputs": [str(item) for item in expected_outputs],
            "progress_files": [str(item) for item in progress_files],
        }
        print()
        print("=" * 96)
        print(f"[{index}/{total} {stage_percent:.1f}%] {stage_label}")
        print("=" * 96)
        if not enabled:
            print("SKIP disabled")
            self.log({**stage_payload, "status": "skipped_disabled"})
            return
        if not force and outputs_complete(self.base_dir, expected_outputs):
            print("SKIP complete: expected outputs already exist")
            snapshots = self.print_snapshots(progress_files)
            self.log({**stage_payload, "status": "skipped_complete", "progress_snapshots": snapshots})
            return
        notebook_path = Path(notebook_dir) / notebook
        if not notebook_path.exists():
            self.log({**stage_payload, "status": "failed_missing_notebook", "path": str(notebook_path)})
            raise FileNotFoundError(f"Notebook not found: {notebook_path}")
        self.log({**stage_payload, "status": "started", "progress_snapshots": progress_snapshots(self.base_dir, progress_files)})
        started = time.time()
        try:
            ipython.run_line_magic("run", str(notebook_path))
        except Exception as exc:
            snapshots = self.print_snapshots(progress_files)
            self.log(
                {
                    **stage_payload,
                    "status": "failed",
                    "elapsed_sec": round(time.time() - started, 3),
                    "error": str(exc),
                    "progress_snapshots": snapshots,
                }
            )
            if _is_transport_endpoint_error(exc):
                print()
                print("DRIVE MOUNT ERROR:", DRIVE_TRANSPORT_HINT)
                raise RuntimeError(DRIVE_TRANSPORT_HINT) from exc
            raise
        snapshots = self.print_snapshots(progress_files)
        expected_status = []
        for item in expected_outputs:
            path = Path(item)
            if not path.is_absolute():
                path = self.base_dir / path
            expected_status.append({"path": str(path), "exists": path.exists()})
            print("expected:", path, "exists=", path.exists())
        self.log(
            {
                **stage_payload,
                "status": "complete",
                "elapsed_sec": round(time.time() - started, 3),
                "progress_snapshots": snapshots,
                "expected_status": expected_status,
            }
        )

    def run_stages(self, *, ipython: Any, notebook_dir: str | Path, stages: list[dict[str, Any]]) -> None:
        total = len(stages)
        for index, stage in enumerate(stages, start=1):
            self.run_stage(
                ipython=ipython,
                notebook_dir=notebook_dir,
                index=index,
                total=total,
                notebook=stage["notebook"],
                label=stage.get("label"),
                enabled=bool(stage.get("enabled", True)),
                force=bool(stage.get("force", False)),
                expected_outputs=stage.get("expected_outputs", ()),
                progress_files=stage.get("progress_files", ()),
            )
        self.log({"status": "compact_run_complete", "stages": total})
        print()
        print("compact log ->", self.log_path)
