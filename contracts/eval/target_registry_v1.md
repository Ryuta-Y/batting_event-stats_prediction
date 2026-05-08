# Target Registry v1

Artifact:

```text
configs/targets/target_registry_v1.yaml
```

This registry is the source of truth for model heads, loss names, metric names, target levels, and PA-manifest requirements. Downstream model code should iterate over registry targets instead of hardcoding only EV / LA / hard-hit / barrel.

The file is JSON-compatible YAML so local tests can load it with the Python standard library. It may be converted to conventional YAML later without changing the contract.

## Event-Level Heads

| target | source | kind | required |
|---|---|---|---|
| `ev` | `launch_speed` | regression | yes |
| `la` | `launch_angle` | regression | yes |
| `hard_hit` | `target_hard_hit`, derived from `launch_speed >= 95` | binary | yes |
| `barrel` | `target_barrel` or documented Statcast EV/LA rule | binary | yes |
| `xba` | `estimated_ba_using_speedangle` | probability | no |
| `xwoba` | `estimated_woba_using_speedangle` | regression | no |

Optional labels must be masked out of loss and official metrics when unavailable. Missing xBA / xwOBA is never zero.

## Aggregate Heads

| target | level | requirement |
|---|---|---|
| `ops` | `player_season` | PA-level manifest |

OPS is not an event-level BBE head. If PA-level data is absent, OPS rows must be skipped with `label_missing_reason=pa_manifest_unavailable`.

