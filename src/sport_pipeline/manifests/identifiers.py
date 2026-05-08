"""Stable identifier helpers for Data Agent A artifacts."""

from __future__ import annotations


def build_batter_season_id(batter_id: int | str, season: int | str) -> str:
    """Build the grouping key for same batter and season."""

    return f"{batter_id}_{season}"


def build_event_id(
    game_pk: int | str,
    at_bat_number: int | str | None,
    pitch_number: int | str | None,
    sv_id: str | None = None,
    play_id: str | None = None,
) -> str:
    """Build a stable BBE event id from Statcast join keys."""

    suffix = play_id or sv_id or "no_play_id"
    return f"game{game_pk}_ab{at_bat_number}_p{pitch_number}_{suffix}"


def build_same_event_group_id(event_id: str) -> str:
    """Default same-event grouping key for views/crops/augmentations."""

    return event_id

