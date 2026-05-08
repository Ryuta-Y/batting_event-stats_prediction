# Target Availability Contract v1

This project keeps unavailable labels explicit. Rows are not dropped just because an optional target is missing.

## BBE-Level Targets

| target | source | availability rule |
|---|---|---|
| `ev` | `launch_speed` | required for official event-level modeling |
| `la` | `launch_angle` | required for official event-level modeling |
| `hard_hit` | `launch_speed >= 95` | available when EV exists |
| `barrel` | Statcast barrel or documented EV/LA derivation | available when source/derivation is available |
| `xba` | `estimated_ba_using_speedangle` | optional; missing is not zero |
| `xwoba` | `estimated_woba_using_speedangle` | optional; missing is not zero |

## PA-Level Required Targets

| target | requirement |
|---|---|
| `ops` | PA-level manifest with OBP and SLG denominators |
| `obp` | PA-level manifest with PA, AB, BB, HBP, SF components |
| `slg` | PA-level manifest with AB and total bases |

When PA-level data is absent:

```text
target_ops_available=false
target_ops_missing_reason=pa_manifest_unavailable
```

Do not create a BBE-only event-level OPS target.

