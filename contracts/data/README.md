# Data Contracts

Phase 2 fixes the data-layer contracts owned by Data Agent A. These files define the expected shape and semantics of the first pipeline artifacts, without implementing large data downloads or downstream modeling.

Official Drive artifact paths are relative to:

```text
/content/drive/MyDrive/baseball_vision
```

Owned artifacts:

- `manifests/bbe_events_v1.parquet`
- `manifests/video_sources_v1.parquet`
- `manifests/splits/player_group_split_v1.parquet`
- `manifests/splits/temporal_split_v1.parquet`

Supporting documentation:

- `contracts/data/bbe_events_v1.md`
- `contracts/data/video_sources_v1.md`
- `contracts/data/splits_v1.md`
- `contracts/data/target_availability_v1.md`
- `docs/contracts/data/AGENT_A_HANDOFF.md`

Machine-readable starting points:

- `configs/data/manifest_contract_v1.json`
- `manifests/templates/bbe_events_v1.sample.jsonl`
- `manifests/templates/video_sources_v1.sample.jsonl`
- `manifests/templates/splits/player_group_split_v1.sample.jsonl`
- `manifests/templates/splits/temporal_split_v1.sample.jsonl`

