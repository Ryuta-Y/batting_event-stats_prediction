# Sequence Contracts

Owner: Agent C Structured Sequence / Player-Season Prior.

Agent C consumes CV metadata from Agent B and produces structured sequence features, clip embeddings, player-season mechanics priors, and event-level datasets conditioned on those priors.

The main flow is:

```text
clips_v1 + CV raw artifacts
  -> features/structured_sequence_v1
  -> features/clip_embedding_v1
  -> features/player_season_embedding_v1
  -> datasets/sequence_dataset_v1
  -> datasets/event_with_player_prior_v1
  -> predictions_v1
```

Large feature arrays and embeddings belong under the Drive artifact root:

```text
/content/drive/MyDrive/baseball_vision
```

This repo stores only contracts, configs, samples, and lightweight validation code.

