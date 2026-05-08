# Sequence Predictions Mapping v1

Agent C outputs must use D1 `predictions_v1` long format.

For event-level structured sequence rows:

```text
prediction_level = event
target_name in [ev, la, hard_hit, barrel, xba, xwoba]
aggregation_scope = current_event_only
prior_mode = none
n_prior_clips = 0
same_event_ensemble = false
```

For event + player-season prior rows:

```text
prediction_level = event
target_name in [ev, la, hard_hit, barrel, xba, xwoba]
aggregation_scope = current_event_with_player_season_prior
prior_mode in [past_only, same_season_train_only, oracle_full_season]
n_prior_clips = count of prior clips used
aggregation_method in [mean_pooling, quality_weighted_pooling, attention_pooling]
same_event_ensemble = false
```

For same-event view/crop/augmentation ensembles:

```text
prediction_level = event
aggregation_scope = same_event_view_crop_augmentation_ensemble
prior_mode = none
same_event_ensemble = true
```

OPS must not be emitted as an event-level prediction. OPS belongs only to player-season or rolling-window aggregate tasks when PA-level data is available.

