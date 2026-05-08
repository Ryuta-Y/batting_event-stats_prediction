"""Small IO helpers for local contract tests and Colab preflights."""

from sport_pipeline.io.jsonl import read_jsonl
from sport_pipeline.io.table import read_table, write_table

__all__ = ["read_jsonl", "read_table", "write_table"]
