# Metrics v1

Artifact:

```text
predictions/{run_id}/metrics_v1.json
```

Owner: Agent D1.

Metrics are computed from `predictions_v1` long-format rows after applying the official availability filter.

## Required Top-Level Fields

- `schema_version`
- `run_id`
- `metrics`
- `label_availability`
- `skipped`

## Official Metric Filter

A row is eligible for official metrics only when:

- `target_available == true`
- `y_true` is not null
- `y_pred` is not null
- `prediction_level` matches the registry target level
- PA-required targets such as OPS have the required PA-level data

Rows failing the filter are counted under `skipped` with a reason. They are not dropped silently.

## Minimum Metrics

Regression / probability:

- `mae`
- `rmse`
- `r2` when requested
- `spearman` when requested

Binary:

- `f1`
- `brier`

Every target entry must include `n_available` and `n_skipped`.

