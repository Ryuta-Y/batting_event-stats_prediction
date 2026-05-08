"""Small table IO helpers for Colab artifacts and local smoke tests."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _normalize_table_value(value: Any) -> Any:
    """Normalize optional dataframe/arrow values into plain Python objects."""

    if isinstance(value, dict):
        return {key: _normalize_table_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_table_value(item) for item in value]
    if value is None:
        return None
    if not isinstance(value, (str, bytes)) and hasattr(value, "tolist"):
        try:
            return _normalize_table_value(value.tolist())
        except Exception:
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    if value.__class__.__name__ in {"NAType", "NaTType"}:
        return None
    try:
        if not isinstance(value, (str, bytes)) and bool(value != value):
            return None
    except Exception:
        pass
    return value


def _normalize_table_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: _normalize_table_value(value) for key, value in row.items()} for row in rows]


def read_table(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL/JSON/CSV/Parquet rows.

    Parquet support is optional and intentionally imported lazily because local
    contract tests should not need pandas or pyarrow. Colab runs are expected to
    have those packages when writing the official parquet artifacts.
    """

    table_path = Path(path)
    suffix = table_path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with table_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        payload = json.loads(table_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return _normalize_table_rows(payload)
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return _normalize_table_rows(payload["rows"])
        raise ValueError(f"JSON table must be a list or contain rows: {table_path}")
    if suffix == ".csv":
        with table_path.open("r", encoding="utf-8", newline="") as handle:
            return _normalize_table_rows(csv.DictReader(handle))
    if suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Reading parquet artifacts requires pandas/pyarrow in Colab") from exc
        return _normalize_table_rows(pd.read_parquet(table_path).to_dict(orient="records"))
    raise ValueError(f"Unsupported table extension: {table_path}")


def write_table(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Write rows to JSONL/JSON/CSV/Parquet based on suffix."""

    table_path = Path(path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    row_list = list(rows)
    suffix = table_path.suffix.lower()
    if suffix == ".jsonl":
        with table_path.open("w", encoding="utf-8") as handle:
            for row in row_list:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return table_path
    if suffix == ".json":
        table_path.write_text(json.dumps(row_list, ensure_ascii=False, indent=2), encoding="utf-8")
        return table_path
    if suffix == ".csv":
        fieldnames: list[str] = []
        for row in row_list:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with table_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(row_list)
        return table_path
    if suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Writing parquet artifacts requires pandas/pyarrow in Colab") from exc
        pd.DataFrame(row_list).to_parquet(table_path, index=False)
        return table_path
    raise ValueError(f"Unsupported table extension: {table_path}")
