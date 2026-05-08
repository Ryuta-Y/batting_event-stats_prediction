# Video Sources Manifest v1

Artifact:

```text
manifests/video_sources_v1.parquet
```

Owner: Data Agent A.

Purpose: candidate video evidence for selected BBE events. A row is a candidate source, not proof that a clean clip exists.

## Required Columns

Identifiers:

- `video_source_id`: stable source row id.
- `event_id`: BBE event id this source is intended to evidence.
- `same_event_group_id`: inherited from the event.
- `source_video_id`: stable id for the media item when available.
- `view_id`: source-level view candidate id. Downstream clips can derive crop or augmentation ids from this.

Source metadata:

- `source_kind`: `mlb_film_room`, `mlb_highlight`, `mlb_video_search`, `local_file`, `manual_reference`, or `other`.
- `source_url`: page URL or source reference.
- `media_url`: direct media URL when known.
- `source_topic`: e.g. `statcast_bbe_search`, `home_runs`, `manual_probe`.
- `dataset_role`: `train_candidate`, `eval_candidate`, `smoke_test`, or `excluded`.
- `rights_status`: `public_domain`, `open_dataset`, `official_public_reference`, `personal_research_only`, or `check_required`.

Match metadata:

- `match_confidence`: numeric confidence from 0 to 1.
- `match_reason`: explanation of the join evidence.
- `join_key_fields`: JSON/list string naming the fields used.
- `candidate_rank`: lower is preferred.

Download and lifecycle:

- `video_available`: candidate exists and is accessible enough to try.
- `download_status`: `not_attempted`, `referenced_only`, `downloaded`, `failed`, or `blocked`.
- `local_video_path`: Drive or cache path if downloaded.
- `probe_status`: `pending`, `ok`, `failed`, or `review_only`.
- `review_status`: `usable_primary`, `review_only`, `excluded`, or `pending`.
- `reject_reason`: null unless excluded.

Quality hints for CV Agent B:

- `view_label`: `batter_side`, `center_field`, `catcher_view`, `broadcast`, `replay`, `unknown`, etc.
- `view_confidence`
- `batting_visibility`: `visible`, `partial`, `not_visible`, or `unknown`.
- `is_replay`: replay/edit flag.
- `is_non_batting_segment`: true when the candidate is clearly not swing evidence.

## Home Run Rule

Home-run topic sources may be retained for smoke tests and failure-browser examples, but must use:

```text
source_topic=home_runs
dataset_role=smoke_test
```

They must not define the training or evaluation population.

