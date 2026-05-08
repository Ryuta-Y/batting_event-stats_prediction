"""Colab-side Statcast/Baseball Savant download helpers.

The downloader is deliberately explicit: it builds a URL plan by default and
only performs network IO when `execute=True` / `--execute` is set. Downloaded
CSV files are saved under the Drive artifact root, then filtered locally to BBE
rows before the existing manifest builder normalizes them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from sport_pipeline.build_manifest import build_manifest_artifacts


BASEBALL_SAVANT_STATCAST_CSV_URL = "https://baseballsavant.mlb.com/statcast_search/csv"


@dataclass(frozen=True)
class StatcastDownloadChunk:
    """One Baseball Savant date chunk."""

    start_date: str
    end_date: str
    url: str
    raw_csv_path: str
    status: str = "planned"
    rows_downloaded: int | None = None
    bbe_rows: int | None = None
    size_bytes: int | None = None
    columns: list[str] | None = None
    source: str = "baseball_savant_statcast_search_csv"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _date_chunks(start_date: str, end_date: str, chunk_days: int) -> list[tuple[str, str]]:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if end < start:
        raise ValueError("end_date must be >= start_date")
    chunks: list[tuple[str, str]] = []
    current = start
    while current <= end:
        chunk_end = min(end, current + timedelta(days=max(chunk_days, 1) - 1))
        chunks.append((current.isoformat(), chunk_end.isoformat()))
        current = chunk_end + timedelta(days=1)
    return chunks


def build_statcast_csv_url(
    start_date: str,
    end_date: str,
    *,
    game_type: str = "R",
    player_type: str = "pitcher",
    only_bbe_hint: bool = True,
) -> str:
    """Build a Baseball Savant CSV URL.

    The local BBE filter remains authoritative because the public query
    parameter names can shift. The URL mirrors pybaseball's conservative
    Statcast date-range request shape and intentionally leaves `hfPR` empty:
    asking the endpoint for every pitch is more stable, then we filter BBE rows
    locally by `type == X` / `description == hit_into_play`.
    """

    params = {
        "all": "true",
        "hfPT": "",
        "hfAB": "",
        "hfBBT": "",
        "hfPR": "",
        "hfZ": "",
        "stadium": "",
        "hfBBL": "",
        "hfNewZones": "",
        "hfGT": f"{game_type}|",
        "hfSea": "",
        "hfSit": "",
        "player_type": player_type,
        "hfOuts": "",
        "opponent": "",
        "pitcher_throws": "",
        "batter_stands": "",
        "hfSA": "",
        "game_date_gt": start_date,
        "game_date_lt": end_date,
        "team": "",
        "position": "",
        "hfRO": "",
        "home_road": "",
        "hfFlag": "",
        "metric_1": "",
        "hfInn": "",
        "min_pitches": "0",
        "min_results": "0",
        "group_by": "name",
        "sort_col": "pitches",
        "player_event_sort": "h_launch_speed",
        "sort_order": "desc",
        "min_abs": "0",
        "type": "details",
    }
    return f"{BASEBALL_SAVANT_STATCAST_CSV_URL}?{urllib.parse.urlencode(params)}"


def plan_statcast_download(
    start_date: str,
    end_date: str,
    output_dir: str | Path,
    *,
    chunk_days: int = 1,
    game_type: str = "R",
) -> list[StatcastDownloadChunk]:
    """Return a dry-run Statcast CSV download plan."""

    root = Path(output_dir)
    chunks = []
    for chunk_start, chunk_end in _date_chunks(start_date, end_date, chunk_days=chunk_days):
        filename = f"statcast_{chunk_start}_to_{chunk_end}.csv"
        chunks.append(
            StatcastDownloadChunk(
                start_date=chunk_start,
                end_date=chunk_end,
                url=build_statcast_csv_url(chunk_start, chunk_end, game_type=game_type),
                raw_csv_path=str(root / filename),
            )
        )
    return chunks


def _download_url(url: str, output_path: Path, timeout_sec: int = 120) -> int:
    request = urllib.request.Request(url, headers={"User-Agent": "sport-pipeline-research/1.0"})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            payload = response.read()
        output_path.write_bytes(payload)
        return len(payload)
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        curl = shutil.which("curl")
        if curl is None:
            raise
        cmd = [
            curl,
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "--max-time",
            str(max(timeout_sec, 1)),
            "-A",
            "sport-pipeline-research/1.0",
            "-o",
            str(output_path),
            url,
        ]
        try:
            subprocess.run(cmd, check=True, timeout=max(timeout_sec + 15, 30))
        except (OSError, subprocess.SubprocessError) as curl_exc:
            raise RuntimeError(f"urllib download failed ({exc}); curl fallback failed ({curl_exc})") from curl_exc
        return output_path.stat().st_size


def _csv_summary(path: str | Path) -> dict[str, Any]:
    csv_path = Path(path)
    columns: list[str] = []
    rows = 0
    bbe_rows = 0
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        for row in reader:
            rows += 1
            if is_bbe_statcast_row(row):
                bbe_rows += 1
    digest = hashlib.sha1(csv_path.read_bytes()).hexdigest() if csv_path.exists() else None
    return {
        "rows": rows,
        "bbe_rows": bbe_rows,
        "columns": columns,
        "size_bytes": csv_path.stat().st_size if csv_path.exists() else 0,
        "sha1": digest,
    }


def _count_csv_rows(path: str | Path) -> int:
    return int(_csv_summary(path)["rows"])


def _has_value(value: Any) -> bool:
    return value is not None and str(value).strip() not in {"", "null", "None", "nan"}


def is_bbe_statcast_row(row: dict[str, Any]) -> bool:
    """Return true when a Statcast row should enter bbe_events_v1."""

    pitch_result = str(row.get("type") or "").strip()
    description = str(row.get("description") or "").strip()
    if pitch_result and pitch_result != "X":
        return False
    if description and description != "hit_into_play":
        return False
    return _has_value(row.get("game_pk")) and _has_value(row.get("batter")) and _has_value(row.get("pitcher"))


def filter_statcast_bbe_rows(input_csv_paths: Iterable[str | Path], output_csv: str | Path) -> dict[str, Any]:
    """Combine raw Statcast CSV chunks and write locally filtered BBE rows."""

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    rows: list[dict[str, Any]] = []
    n_raw = 0
    for input_path in input_csv_paths:
        with Path(input_path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for name in reader.fieldnames or []:
                if name not in fieldnames:
                    fieldnames.append(name)
            for row in reader:
                n_raw += 1
                if is_bbe_statcast_row(row):
                    rows.append(row)
    if not fieldnames and rows:
        for row in rows:
            for name in row:
                if name not in fieldnames:
                    fieldnames.append(name)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "raw_rows": n_raw,
        "bbe_rows": len(rows),
        "output_csv": str(output),
    }


def _write_progress(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def download_statcast_bbe_dataset(
    base_dir: str | Path,
    start_date: str,
    end_date: str,
    *,
    execute: bool = False,
    chunk_days: int = 1,
    game_type: str = "R",
    build_manifest: bool = True,
    output_suffix: str = ".parquet",
    timeout_sec: int = 120,
    resume: bool = True,
    redownload_empty: bool = True,
    allow_empty_manifest: bool = False,
    allow_partial_manifest: bool = False,
) -> dict[str, Any]:
    """Plan or execute Statcast download, BBE filtering, and manifest build."""

    base = Path(base_dir)
    raw_dir = base / "raw_statcast"
    progress_path = raw_dir / f"statcast_download_progress_{start_date}_to_{end_date}.json"
    summary_path = raw_dir / f"statcast_download_summary_{start_date}_to_{end_date}.json"
    plan = plan_statcast_download(
        start_date=start_date,
        end_date=end_date,
        output_dir=raw_dir,
        chunk_days=chunk_days,
        game_type=game_type,
    )
    chunk_payloads: list[dict[str, Any]] = []
    raw_paths: list[Path] = []
    progress_payload: dict[str, Any] = {
        "schema_version": "statcast_download_progress_v1",
        "execute": execute,
        "start_date": start_date,
        "end_date": end_date,
        "chunk_days": chunk_days,
        "game_type": game_type,
        "resume": resume,
        "redownload_empty": redownload_empty,
        "allow_partial_manifest": allow_partial_manifest,
        "chunks": chunk_payloads,
    }
    for chunk in plan:
        payload = chunk.to_dict()
        path = Path(chunk.raw_csv_path)
        if not execute:
            chunk_payloads.append(payload)
            _write_progress(progress_path, progress_payload)
            continue
        try:
            should_download = True
            existing_summary: dict[str, Any] | None = None
            if resume and path.exists():
                existing_summary = _csv_summary(path)
                existing_rows = int(existing_summary["rows"])
                if existing_rows > 0 or not redownload_empty:
                    should_download = False
                    payload["status"] = "cached" if existing_rows > 0 else "cached_empty"
                    payload["rows_downloaded"] = existing_rows
                    payload["bbe_rows"] = int(existing_summary["bbe_rows"])
                    payload["size_bytes"] = int(existing_summary["size_bytes"])
                    payload["columns"] = list(existing_summary["columns"])
            if should_download:
                previous_empty = bool(path.exists() and existing_summary and int(existing_summary["rows"]) == 0)
                _download_url(chunk.url, path, timeout_sec=timeout_sec)
                downloaded_summary = _csv_summary(path)
                payload["status"] = "redownloaded" if previous_empty else "downloaded"
                payload["rows_downloaded"] = int(downloaded_summary["rows"])
                payload["bbe_rows"] = int(downloaded_summary["bbe_rows"])
                payload["size_bytes"] = int(downloaded_summary["size_bytes"])
                payload["columns"] = list(downloaded_summary["columns"])
            raw_paths.append(path)
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            payload["status"] = "failed"
            payload["error"] = str(exc)
        chunk_payloads.append(payload)
        completed = len(chunk_payloads)
        progress_payload["completed_chunks"] = completed
        progress_payload["total_chunks"] = len(plan)
        progress_payload["raw_rows_so_far"] = sum(int(item.get("rows_downloaded") or 0) for item in chunk_payloads)
        progress_payload["bbe_rows_so_far"] = sum(int(item.get("bbe_rows") or 0) for item in chunk_payloads)
        _write_progress(progress_path, progress_payload)

    manifest_outputs: dict[str, str] = {}
    filter_summary: dict[str, Any] | None = None
    warnings: list[str] = []
    filtered_csv = raw_dir / f"statcast_bbe_{start_date}_to_{end_date}.csv"
    failed_chunks = [item for item in chunk_payloads if item.get("status") == "failed"]
    if execute and raw_paths:
        filter_summary = filter_statcast_bbe_rows(raw_paths, filtered_csv)
        if build_manifest and failed_chunks and not allow_partial_manifest:
            warnings.append(
                f"{len(failed_chunks)} Statcast chunks failed; skipped manifest build to avoid partial event universe"
            )
        elif build_manifest and (allow_empty_manifest or int(filter_summary.get("bbe_rows") or 0) > 0):
            outputs = build_manifest_artifacts(
                base_dir=base,
                input_path=filtered_csv,
                output_suffix=output_suffix,
            )
            manifest_outputs = {key: str(value) for key, value in outputs.items()}
        elif build_manifest:
            warnings.append(
                "filtered BBE rows are 0; skipped manifest build to avoid overwriting existing manifests with empty artifacts"
            )

    summary = {
        "schema_version": "statcast_download_summary_v1",
        "execute": execute,
        "start_date": start_date,
        "end_date": end_date,
        "chunk_days": chunk_days,
        "game_type": game_type,
        "chunks": chunk_payloads,
        "filtered_bbe_csv": str(filtered_csv),
        "filter_summary": filter_summary,
        "manifest_outputs": manifest_outputs,
        "progress_path": str(progress_path),
        "resume": resume,
        "redownload_empty": redownload_empty,
        "allow_empty_manifest": allow_empty_manifest,
        "allow_partial_manifest": allow_partial_manifest,
        "failed_chunks": failed_chunks,
        "warnings": warnings,
        "summary_path": str(summary_path),
    }
    _write_progress(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or execute Baseball Savant Statcast BBE downloads.")
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/baseball_vision")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--chunk-days", type=int, default=1)
    parser.add_argument("--game-type", default="R")
    parser.add_argument("--no-build-manifest", action="store_true")
    parser.add_argument("--output-format", choices=("parquet", "jsonl", "json", "csv"), default="parquet")
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--keep-empty-cache", action="store_true")
    parser.add_argument("--allow-empty-manifest", action="store_true")
    parser.add_argument("--allow-partial-manifest", action="store_true")
    args = parser.parse_args()
    result = download_statcast_bbe_dataset(
        base_dir=args.base_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        execute=args.execute,
        chunk_days=args.chunk_days,
        game_type=args.game_type,
        build_manifest=not args.no_build_manifest,
        output_suffix="." + args.output_format,
        timeout_sec=args.timeout_sec,
        resume=not args.no_resume,
        redownload_empty=not args.keep_empty_cache,
        allow_empty_manifest=args.allow_empty_manifest,
        allow_partial_manifest=args.allow_partial_manifest,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
