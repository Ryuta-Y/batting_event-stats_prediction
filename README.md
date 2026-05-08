# MLB Batting Vision v2

MLB の打撃動画 clip と Statcast BBE データを結び付け、打球指標と選手シーズン成績を予測・比較する Google Colab / Google Drive 用の研究パイプラインです。

このフォルダは GitHub 公開用に整理したコピーです。元の作業フォルダは変更していません。

## What This Repository Contains

この公開用コピーは、v2 実験の本筋だけに絞っています。

- `notebooks/30_cpu_data_sources_labels_reuse.ipynb`
- `notebooks/31_cpu_context_cv_sequence_light_video.ipynb`
- `notebooks/32_gpu_deep_cv_feature_extraction.ipynb`
- `notebooks/33_gpu_sequence_and_video_models.ipynb`
- `notebooks/34_gpu_vlm_mechanics.ipynb`
- `notebooks/35_cpu_evaluation_fusion_reports.ipynb`
- 上記 wrapper が内部で呼ぶ thin stage notebooks
- 再利用ロジック: `src/sport_pipeline/`
- v2 run profile: `configs/runs/mlb_2024_2026_real_colab_v2.json`
- target / model / CV / fusion configs
- artifact contracts: `contracts/`
- 完成済み研究レポート: `reports/v2_research_deep_dive/deep_research_document_ja.html`

raw video、Drive上の大きなparquet、model weights、Colab実行時cacheは含めません。

## Fixed Runtime Paths

Colab では次のパスを前提にしています。

```text
Colab code root: /content/drive/MyDrive/codex/batting_codex_handoff
Drive artifact root: /content/drive/MyDrive/baseball_vision
Colab cache: /content/cache/baseball_vision
Run profile: mlb_2024_2026_real_colab_v2.json
Python package: src/sport_pipeline/
```

このリポジトリを Colab で使う場合は、フォルダ名を `batting_codex_handoff` として Drive の次の場所に置くのが一番安全です。

```text
/content/drive/MyDrive/codex/batting_codex_handoff
```

別の場所に置く場合は、notebook 実行前に次を設定してください。

```python
%env BATTING_CODE_ROOT=/content/drive/MyDrive/your/path/batting_codex_handoff
%env BASEBALL_VISION_RUN_PROFILE=mlb_2024_2026_real_colab_v2.json
```

## Pipeline Summary

この研究では、home run clip だけを母集団にしません。先に Statcast の BBE event universe を作り、あとから動画 evidence を紐づけます。

```text
Statcast BBE events
  -> video source resolution / reuse
  -> contact-aligned clips
  -> CV artifacts / video embeddings / VLM mechanics features
  -> per-method event predictions
  -> event-to-season projections
  -> player-season prior predictions
  -> late fusion
  -> method evaluation and research report
```

重要なルール:

- 1 BBE event は 1 event-level prediction として扱う
- 異なる BBE event の予測を平均して、ある1打席の EV / LA 予測にはしない
- 同一 event の複数 view / crop / augmentation だけは ensemble してよい
- 同一 batter-season の複数 clean clips は player-season mechanics prior に使う
- xBA / xwOBA の欠損は 0 埋めしない
- OPS / OBP / SLG / BA は event-level head ではなく player-season target
- BA / OBP / SLG / OPS は予測時の入力特徴ではなく、学習・calibration・評価の教師ラベル

## Main Colab Order

v2 の本筋は次の順に実行します。

| order | notebook | runtime | role |
|---:|---|---|---|
| 30 | `30_cpu_data_sources_labels_reuse.ipynb` | CPU / network | env/init, v2 isolation, v1 reuse, player-season labels, readiness |
| 31 | `31_cpu_context_cv_sequence_light_video.ipynb` | CPU / high RAM | context CatBoost, contact clip preprocessing, structured baseline, lightweight video |
| 32 | `32_gpu_deep_cv_feature_extraction.ipynb` | GPU L4/A100 | YOLO/tracking/pose/bat/plate/homography, sequence features, overlays |
| 33 | `33_gpu_sequence_and_video_models.ipynb` | GPU L4/A100 | TCN, frozen VideoMAE, raw video R3D-18 fine-tune |
| 34 | `34_gpu_vlm_mechanics.ipynb` | GPU L4/A100 | VLM template, Qwen VLM caption/tags/scores, VLM baseline |
| 35 | `35_cpu_evaluation_fusion_reports.ipynb` | CPU | player-season projections, fusion, ablation, method evaluation, research outputs |

The wrapper notebooks call these supporting stage notebooks:

```text
00_check_env, 01_init_drive,
05b_context_catboost_baseline,
09_report_builder,
10_full_run_readiness, 10b_run_isolation_check,
11_download_statcast_and_video_sources,
11_a/b/c_download_video_shard, 11_d_merge_video_download_shards,
11_e_seed_v2_download_manifest_from_v1,
11_f_download_player_season_batting_stats,
12_full_cv_preprocess,
13_full_sequence_baseline,
14_full_video_baseline,
15_full_fusion,
16_deep_cv_yolo_pose_homography,
17_deep_sequence_features,
18_sequence_tcn_training,
19_frozen_visual_encoder,
19b_raw_video_finetune,
20_video_ablation_compare,
21_cv_overlay_videos,
22_research_outputs,
23_player_season_aggregate_baseline,
24_vlm_mechanics_baseline,
24b_hf_vlm_captioning,
25_method_evaluation,
26_event_method_player_season_projection
```

## Outputs

Colab 実行で作られる大きな成果物は、すべて Drive root に保存されます。

```text
/content/drive/MyDrive/baseball_vision
  manifests/
  raw_videos/
  clips/
  detections/
  tracks/
  pose2d/
  objects/
  homography/
  features/
  datasets/
  models/
  predictions/
  reports/
  debug/
```

最終的に見るべき研究レポートは次です。

```text
reports/v2_research_deep_dive/deep_research_document_ja.html
```

主要な図:

- `reports/v2_research_deep_dive/assets/diagrams/method_flow_corrected_v3.svg`
- `reports/v2_research_deep_dive/assets/diagrams/notebook_31_35_artifact_detail.svg`
- `reports/v2_research_deep_dive/assets/diagrams/event_same_sample_rank_heatmap.svg`
- `reports/v2_research_deep_dive/assets/diagrams/player_same_sample_spearman_heatmap.svg`

## Model Families

| method | input | model / processing | prediction level |
|---|---|---|---|
| Context CatBoost | Statcast context columns | CatBoost tabular baseline | event, projected season |
| OpenCV lightweight | contact clip pixels | RGB / motion statistics | event, projected season |
| Frozen visual encoder | contact clip frames | VideoMAE frozen embedding + supervised head | event, projected season |
| Raw video fine-tune | contact clip frames | torchvision R3D-18 style 3D CNN | event, projected season |
| CV sequence / TCN | detections, tracking, pose, bat line | structured sequence + TCN | event, projected season |
| VLM mechanics | selected clip videos | Qwen2.5-VL captions/tags/scores + lightweight head | event, projected season |
| Player-season prior | multiple clips per batter-season | mechanics prior aggregate baseline | player-season |
| Late fusion | upstream prediction rows | fixed weighted average by sample and target | event and player-season |

## Current v2 Scale

The completed v2 run used approximately:

- 272,989 Statcast BBE events
- 930 downloaded / reused videos
- 814 contact clips
- 1,180,924 detection rows
- 110,203 pose rows
- 687 structured/video model clips
- 100 complete VLM rows out of 687 template rows
- 1,768 player-season samples for the broad player-season comparison

See the report for exact per-method sample counts and same-sample comparisons.

## Dependencies

The notebooks install or import dependencies inside Colab as needed. Heavy dependencies and model downloads are intentionally opt-in.

Typical packages:

- Python 3.12 in Colab
- pandas / pyarrow
- numpy / scikit-learn
- torch / torchvision
- transformers / accelerate
- ultralytics
- mediapipe
- catboost
- opencv-python
- qwen-vl-utils / decord for VLM video reading

Do not run heavy training, full video processing, or large model downloads locally. The local repository is for code/config/report artifacts; the heavy work belongs in Colab.

## Repository Layout

```text
configs/              v2 run profile and model/stage settings
contracts/            artifact schemas and prediction contracts
notebooks/            Colab entrypoints
src/sport_pipeline/   reusable Python implementation
reports/v2_research_deep_dive/
                      final v2 research report and figures
```

## What Was Removed From This GitHub Copy

This cleaned copy intentionally removes items that were useful during development but are not part of the v2 30-35 research path:

- old design handoff documents
- reference Colab example scripts
- sample-index preview materials
- local video audit artifacts
- smoke-only notebooks
- legacy v1/smoke run profiles
- standalone YOLO object-detector training helper
- audio branch notebooks and code
- local unit tests
- Python bytecode caches and macOS metadata
- old duplicate report folder

The original full working folder was not modified.

## Known Limitations

- The VLM branch is exploratory: only 100 of 687 VLM template rows were completed in the reported v2 run.
- The pose branch may sometimes track the pitcher or non-batter body, so batter association should be improved in the next version.
- The final report is a static artifact. It summarizes the completed v2 run but does not include the large Drive data required to reproduce every number from scratch.
- GitHub should not contain raw MLB videos, model weights, or Drive-generated large artifacts.
