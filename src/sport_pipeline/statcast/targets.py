"""Target ownership rules for Statcast and PA-level labels."""

BBE_ONLY_TARGETS = {
    "ev": "launch_speed",
    "la": "launch_angle",
    "hard_hit": "launch_speed >= 95",
    "barrel": "statcast_or_documented_ev_la_rule",
    "xba": "estimated_ba_using_speedangle",
    "xwoba": "estimated_woba_using_speedangle",
}

PA_REQUIRED_TARGETS = {
    "ops": "target_ops",
    "obp": "target_obp",
    "slg": "target_slg",
}


def ops_missing_reason(has_pa_manifest: bool) -> str | None:
    """Return the official OPS missing reason for BBE-only data."""

    if has_pa_manifest:
        return None
    return "pa_manifest_unavailable"

