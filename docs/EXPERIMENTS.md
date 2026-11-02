# Experiment Log, Hyperparameters, and Tuning Plan

All experiments run on 2026-11-02 unless stated otherwise.

Environment: Python 3.10.11, torch 2.11.0+cu128, torchvision 0.26.0+cu128,
Anomalib 2.7.0, RTX 3060 12 GB, 7.9 GB system RAM.

---

## 1. Evaluation protocol (current)

### Rules

- Fit on normal training data only.
- Evaluate on the test split.
- Report frame/image AUROC as the threshold-free main metric.
- Report pixel AUROC and pixel F1 for localization.
- Threshold for F1 comes from Anomalib's adaptive threshold on the validation split.
- Record the exact sampling protocol; never compare numbers across different protocols.

### Current memory-safe compromise

Because of the 7.9 GB RAM limit:

- **Training memory bank:** every `N`th normal frame (configurable; first run N=100).
- **Test set:** every `M`th frame for pixel metrics (first real run M=10).
- Image-level AUROC on all test frames is cheap; pixel AUROC on all frames is not.

The final benchmark should replace M with a streaming/per-video metric so that all test
frames are evaluated without materialising all pixel predictions at once.

---

## 2. Experiment E-000 — Smoke tests

### E-000a — Avenue data pipeline

Command (conceptual):

```python
dm = Avenue(root="data/avenue", gt_dir="data/avenue/ground_truth_demo",
            clip_length_in_frames=1, batch_size=2, num_workers=0)
dm.prepare_data(); dm.setup()
```

Result:

- train samples: 15,328
- test samples: 15,324
- image batch: `(2, 3, 480, 856)`

Purpose: confirm the datamodule and preprocessing work.

### E-000b — PatchCore limited fit/test

Configuration:

- scene 1, clip length 1
- batch size 8
- `limit_train_batches=5`, `limit_val_batches=2`, `limit_test_batches=2`
- PatchCore, WRN50, layers 2+3, coreset 0.01, neighbors 9
- `visualizer=False`

Result:

- Fit: success, GPU used, coreset selected over 408 candidates.
- Test: success after disabling the visualizer.
- Metrics: 0.0 because the limited test subset had no anomalous frames.

Purpose: verify GPU training/inference and identify the visualizer bug.

### E-000c — Full-test failure

Configuration:

- Same fitted model.
- Test datamodule stride 1 (11,894 frames).

Result:

- Inference completed (744 batches, 33 min 52 s).
- AUROC computation failed allocating 6.2 GB.

Purpose: establish the RAM ceiling; not a valid result.

---

## 3. Experiment E-001 — PatchCore, ShanghaiTech scene 1

### Status

Completed (subsampled protocol).

### Configuration

| Item | Value |
|---|---|
| Model | PatchCore |
| Backbone | `wide_resnet50_2` |
| Layers | `layer2`, `layer3` |
| Pretrained | yes |
| Input size | 256×256 |
| Coreset ratio | 0.01 |
| Neighbors | 9 |
| Precision | float32 |
| Train sampling | every 100th normal frame |
| Train clips | ~700 (47 batches × 16) |
| Test sampling | every 10th test frame |
| Test clips | 1,204 |
| Train batch size | 16 |
| Eval batch size | 16 |
| Workers | 0 |
| Max epochs | 1 |
| Visualizer | disabled |
| Threshold | adaptive from validation |
| Seed | not fixed |

### Timings

| Stage | Time |
|---|---|
| Fit (including feature extraction + coreset) | 239.5 s |
| Test inference + metrics (1,204 frames) | 379.9 s |
| Full-test inference before OOM (11,894 frames) | 2,032 s |

### Results

| Metric | Value |
|---|---|
| image_AUROC | 0.78525 |
| image_F1Score | 0.67164 |
| pixel_AUROC | 0.98871 |
| pixel_F1Score | 0.59476 |

### Artifacts

```
results/patchcore/scene01/Patchcore/ShanghaiTech/v0/weights/lightning/model.ckpt
```

### Observations

- Pixel localization is much stronger than frame-level detection.
- A frame-wise model likely misses motion-only anomalies (running, chasing).
- The memory bank is small (~700 frames), which may limit coverage of normal variation.
- Test sampling changes the metric; these numbers are not directly comparable to published
  full-test results.

### What this experiment does **not** establish

- Full-test performance.
- Cross-scene generalization.
- A fair head-to-head against video-native methods.
- The effect of hyperparameters.

---

## 3.5 Experiment E-002 — Annotated demo video (visualization)

### Status

Completed.

### Goal

Produce a watchable MP4 with red boxes around detected anomalies, using the E-001
PatchCore checkpoint.

### Source clip

- ShanghaiTech scene 1, clip `01_0136`
- 529 frames, 856×480
- Ground-truth frame mask: 311 anomalous frames (first positive index 133)

### Renderer

`src/render_video.py`, loading:

```
results/patchcore/scene01/Patchcore/ShanghaiTech/v0/weights/lightning/model.ckpt
```

Pipeline per frame:

1. Read JPEG (BGR) → RGB → float/255.
2. Apply the checkpoint's pre-processor: resize 256×256 + ImageNet normalize.
3. Forward pass through the PatchCore torch model.
4. Threshold the 256×256 anomaly map, morphological open/close.
5. Upscale the mask to 856×480, find contours, filter by area, keep top-k.
6. Draw red rectangles; draw a banner with `ANOMALY`/`NORMAL`, raw score, box count.
7. Write at 10 fps.

### Display settings

| Setting | Value | Note |
|---|---|---|
| Pixel threshold | 55.0 | checkpoint value is 48.84 |
| Image threshold | 58.0 | checkpoint value is 46.27 |
| Min box area | 200 px | at 856×480 |
| Top-k boxes | 3 | per frame |
| Boxes drawn only if frame is anomalous | yes | prevents boxes on NORMAL frames |
| FPS | 10 | ShanghaiTech test frames have no fps metadata |

### Why the display thresholds differ from the checkpoint

The checkpoint thresholds produced many small boxes on normal frames (for example frames
62 and 80 on a backpack). A threshold sweep on this clip gave:

| Image threshold | Precision | Recall | F1 |
|---|---|---|---|
| 46.3 (checkpoint) | 0.66 | 1.00 | 0.80 |
| 50 | 0.79 | 0.97 | 0.87 |
| 52 | 0.89 | 0.93 | 0.91 |
| 54 | 0.97 | 0.89 | 0.93 |
| 55 | 0.97 | 0.87 | 0.92 |
| 58 | 0.99 | 0.79 | 0.88 |
| 60 | 0.99 | 0.70 | 0.82 |

Pixel threshold 55 removed the small false boxes while keeping the truck box.

At image threshold 58 the two remaining "false positives" are frames 271 and 323. Visual
inspection shows the **truck is clearly present** in both, while the frame-level GT mask is
0. This is likely sparse annotation rather than a model error, and is recorded as an
error-analysis observation.

### Results

| Item | Value |
|---|---|
| Frames rendered | 529 |
| Render time | 19.8 s (~26.7 fps) |
| Frames labelled ANOMALY | 247 |
| Frames with boxes | 245 |
| First / last boxed frame | 169 / 431 |
| Max raw score | 82.35 (truck) |
| Video size | 4.4 MB |
| Resolution / fps | 856×480 @ 10 fps |

### Artifacts

```
results/videos/shanghaitech_scene01_01_0136_annotated.mp4
results/videos/shanghaitech_scene01_01_0136_annotated.scores.csv
results/videos/01_0136_samples/frame_007.jpg   (NORMAL, no box)
results/videos/01_0136_samples/frame_169.jpg   (first boxed frame)
results/videos/01_0136_samples/frame_297.jpg   (peak truck box, score 82.3)
results/videos/01_0136_samples/frame_309.jpg   (high-confidence box)
```

### Caveats

- Display thresholds are tuned for presentation on this clip; the reported metrics use the
  checkpoint thresholds and the E-001 protocol.
- Frame-wise detection still misses low-score early/late anomaly frames.
- The demo intentionally focuses on the clear truck event.

---

## 3.6 Experiment E-003 — AI-VAD people-only demo video

### Why this experiment exists

The frame-wise PatchCore model was tested on the fight clip `01_0026` (two people
brawling, GT span 278-397). Scores stayed at 44-56 and produced **no boxes**. A scan of all
34 scene-1 clips showed every high-scoring clip was a vehicle (van, car, three-wheeler).
The model is blind to motion anomalies. A video-native model is required.

### Model and fit

| Item | Value |
|---|---|
| Model | `AiVad` |
| `persons_only` | True |
| `enable_foreground_detections` | False |
| `pre_processor` / `post_processor` | False |
| Data | ShanghaiTech scene 1, `clip_length_in_frames=2`, `frames_between_clips=100` |
| Train clips | 738 |
| Batch size | 2 |
| Workers | 0 |
| Max epochs | 1 (density estimation, no gradient training) |
| Validation | disabled (`limit_val_batches=0`) |
| Components | Mask R-CNN, Keypoint R-CNN, RAFT optical flow, CLIP ViT-B/16, velocity/pose/deep density estimators |
| Fit time | **5 min 02 s** |
| Person detections | **2,262** |
| Checkpoint | `results/aivad/scene01/AiVad/ShanghaiTech/v0/weights/lightning/model.ckpt` (762.6 MB) |

### Clip and calibration

- Clip: `01_0026`, 505 frames, 856×480.
- GT frame mask: 120 positive frames, span 278-397.
- A score-only pass over 504 frame pairs took **3.7 min**.
- Raw image scores: normal frames around **1.04**, fight frames **1.41-1.63**, max 2.45.
- Frame-level separation is noisy because normal crowd motion also produces people regions.
- Temporal smoothing (window 15) improves frame-level F1 to **0.73** at threshold 1.2.
- Remaining false positives are concentrated **outside** the fight:
  frames 23-35, 141-202, and 398-403.
- Display choices: **smooth window 15**, **image threshold 1.3**, **box threshold 1.1**,
  **min area 200**, **top-k 3**, cropped to frames **260-420**.

### Render result

| Item | Value |
|---|---|
| Processed pairs | 176 (frames 245-420 for warmup) |
| Output frames | 161 (frames 260-420) |
| Render time | 169.3 s (~1.0 fps) |
| Frames labelled ANOMALY | 101 |
| Frames with boxes | 91 |
| First / last boxed frame | 311 / 405 |
| Max smoothed score | 2.40 |
| Output | 856×480 @ 10 fps, 1.1 MB |

### Artifacts

```
results/videos/shanghaitech_scene01_01_0026_people_annotated.mp4
results/videos/shanghaitech_scene01_01_0026_people_annotated.scores.csv
results/videos/01_0026_people_samples/
```

### Notes and bugs

- **Empty-region bug:** AI-VAD raises
  `IndexError: amax(): Expected reduction dim 0 to have non-zero size` on frames with no
  person detections. The renderer catches this and treats the frame as normal
  (score 1.0, empty map).
- **Causal smoothing:** the moving average lags and tails, so the banner requires at least
  one box to say `PEOPLE ANOMALY`.
- **Region-score correction:** the first renderer ranked each person by the maximum
  anomaly-map value inside its box. The corrected renderer now uses the density estimator's
  per-region `box_scores`, which removes the earlier walker box at frame 263 while retaining
  the fight event.
- **Display thresholds** are tuned for presentation, not a benchmark protocol.
- **GT sparsity / crowd motion** produce false positives outside the fight segment, which is
  why the demo is cropped to the event.
- This is a visualization result. It does not change the metric protocol.

### E-003b — Gap-5 cleaned people-only render

The original consecutive-frame demo was replaced with a deployment-like five-frame gap
(0.5 seconds at 10 FPS), five-frame smoothing, and track persistence. Thresholds were
raised to image **2.25** and box **2.0**; tracks required six qualifying hits in a
ten-frame window, with a 1.7 stop threshold and 20-frame cooldown.

| Item | Value |
|---|---|
| Output | `results/videos/shanghaitech_scene01_01_0026_people_gap5_clean.mp4` |
| Frames / format | 500 frames, 856×480 @ 10 FPS |
| Boxed frames | 33 |
| Boxed ranges | 168-200 and 383-395 |
| Max smoothed score | 4.0447 |
| Score log | `results/videos/shanghaitech_scene01_01_0026_people_gap5_clean.scores.csv` |

This render removes the earlier isolated walking boxes around seconds 6-8, but it is
still a presentation result rather than a retrained gap-matched model evaluation.

---

## 4. Current hyperparameter table

### PatchCore

| Hyperparameter | Current | Typical range | Notes |
|---|---|---|---|
| backbone | wide_resnet50_2 | resnet50, WRN50, ViT | accuracy vs speed |
| layers | layer2+3 | layer2, layer3, layer2+3 | resolution vs semantics |
| coreset_sampling_ratio | 0.01 | 0.001–0.1 | bank size vs accuracy |
| num_neighbors | 9 | 1–9 | score smoothing |
| precision | float32 | float16, float32 | speed vs exactness |
| image_size | 256 | 224–320 | memory and detail |
| memory-bank frames | ~700 | 400–2,000 | biggest accuracy lever |
| test sampling | every 10th | 1–10 | metric memory |

### PaDiM (planned)

| Hyperparameter | Planned initial | Alternatives |
|---|---|---|
| backbone | wide_resnet50_2 | resnet50, ViT |
| layers | layer2+3 | layer2, layer3 |
| n_features | 100 | 550 |
| covariance regularization | default in Anomalib | 0.01, 0.1, 0.5 |
| image_size | 256 | 224, 320 |
| memory-bank frames | ~700 | 400–2,000 |

### AI-VAD (planned)

Constructor parameters verified in Anomalib 2.7:

| Parameter | Purpose |
|---|---|
| `box_score_thresh` | person-detector confidence threshold |
| `persons_only` | restrict to person regions |
| `min_bbox_area` | ignore tiny boxes |
| `max_bbox_overlap` | merge overlapping boxes |
| `enable_foreground_detections` | foreground segmentation |
| `foreground_kernel_size` | morphology kernel |
| `foreground_binary_threshold` | foreground threshold |
| `n_velocity_bins` | motion histogram bins |
| `use_velocity_features` | motion features on/off |
| `use_pose_features` | pose features on/off |
| `use_deep_features` | deep appearance features on/off |
| `n_components_velocity` | PCA components for velocity |
| `n_neighbors_pose` | kNN for pose |
| `n_neighbors_deep` | kNN for deep features |

Initial plan: defaults, clip length ≥2, then tune `box_score_thresh`, `persons_only`,
`min_bbox_area`, and the feature toggles.

### FUVAS (planned)

| Parameter | Purpose |
|---|---|
| `backbone` | feature extractor |
| `layer` | which layer |
| `spatial_pool` | pooling strategy |
| `pooling_kernel_size` | pooling size |
| `pca_level` | PCA reduction |
| few-shot `k` | 1, 2, 4, 8 |

### WinCLIP (planned)

| Parameter | Purpose |
|---|---|
| `class_name` | object class for prompts |
| `k_shot` | 0 for zero-shot; >0 for few-shot |
| `scales` | window scales |
| `few_shot_source` | reference images for few-shot |

Initial plan: zero-shot, scales 2, then prompt/scale ablation.

---

## 5. Tuning plan (priority order)

### T1 — Memory-bank size (highest impact)

- Values: 400, 700, 1,000, 2,000 normal frames per scene.
- Fixed: model/backbone/coreset/test protocol.
- Measure: image AUROC, pixel AUROC, fit time, peak RAM.
- Goal: find the knee where more frames stop helping.

### T2 — Coreset ratio

- Values: 0.001, 0.005, 0.01, 0.05, 0.1.
- Fixed: memory-bank size.
- Measure: AUROC vs memory bank size and test FPS.
- Goal: maximise accuracy per stored vector.

### T3 — Backbone

- Values: `resnet50`, `wide_resnet50_2`, a ViT/DINOv2 feature extractor.
- Fixed: best memory-bank size and coreset.
- Measure: AUROC, fit/test time, VRAM.
- Goal: test whether stronger features compensate for less data.

### T4 — Feature layers

- Values: layer2, layer3, layer2+3.
- Measure: localization (pixel AUROC) vs semantic detection (image AUROC).

### T5 — Input resolution

- Values: 224, 256, 320.
- Measure: AUROC vs inference speed and pixel-metric RAM.
- Note: 320 may exceed the safe RAM budget for full pixel metrics.

### T6 — Neighbors

- Values: 1, 5, 9.
- Measure: AUROC and runtime.

### T7 — Test-evaluation protocol

- Option A: every frame for image AUROC; every 10th for pixel AUROC.
- Option B: streaming per-video AUROC for both.
- Goal: remove sampling bias from the final numbers.

### T8 — PaDiM-specific

- `n_features`: 100 vs 550.
- Covariance regularization: 0.01 / 0.1 / 0.5.
- Compare against PatchCore at matched memory-bank size.

### T9 — AI-VAD

- Enable/disable velocity, pose, and deep features.
- Tune `box_score_thresh`, `min_bbox_area`, `persons_only`.
- Compare with frame-wise models on motion-heavy scenes.

### T10 — FUVAS

- Few-shot curve: k = 1, 2, 4, 8.
- Backbone/layer/PCA level.

### T11 — WinCLIP

- Prompt templates (normal vs abnormal).
- Scales 1 / 2 / 3.
- Zero-shot vs few-shot.

### T12 — Temporal layer

- Smoothing window: 1, 5, 11, 21.
- Moving average vs median vs max.
- Hysteresis and debounce.
- Measure false alarms per minute and detection delay.

### T13 — AI-VAD product-hardening protocol

This is the next controlled experiment rather than a new model benchmark:

1. Fit AI-VAD with deployment-like gaps (2, 5, and 10 frames), not only the original
   stride-100 preparation.
2. Score the same normal-walking and fight clips with matching inference gaps.
3. Compare raw image score, velocity, pose, deep, and per-region score distributions.
4. Apply IoU track histories with a rolling median, six-of-ten persistence, hysteresis,
   debounce, and cooldown.
5. Select thresholds from held-out normal footage at a target false-alert rate.
6. Mine false-positive walking/crowd clips and refit, then repeat the measurement.

The renderer exposes the first implementation:

```powershell
.\.venv\Scripts\python.exe src\render_people_video.py `
  --clip 01_0026 --frame-gap 5 --track-window 10 --track-min-hits 6 `
  --track-stop-hits 10 --cooldown-frames 30
```

Held-out calibration is separate from the fight demo:

```powershell
.\.venv\Scripts\python.exe src\calibrate_people_thresholds.py `
  --frames-dir data\shanghaitech\testing\frames\NORMAL_CLIP `
  --frame-gap 5 --output results\calibration\normal.json
```

The calibration output records the 99.5th percentile image, region, velocity, pose, and
deep thresholds. It must be generated from normal footage not used to fit the density
estimators.

### T13a — Initial held-out calibration and hard-negative screening

The existing scene-1 checkpoint was evaluated without retraining at gaps 2, 5, and 10
on normal training AVIs from scenes 2 and 3. The provisional 99.5th-percentile
thresholds were:

| Gap | Image | Region |
|---:|---:|---:|
| 2 | 2.376 | 1.842 |
| 5 | 2.937 | 2.253 |
| 10 | 5.797 | 3.167 |

These characterize inference-gap sensitivity only; they are not final deployment
thresholds because the checkpoint was fitted with a different temporal preparation.

To avoid treating an entire test clip as normal, candidate hard negatives were restricted
to visually reviewed, mask-zero intervals from `01_0025`, `01_0063`, and `07_0048`.
At gap 5, 36 sampled pairs produced image max 4.288 and region max 3.050, above the
scene-2/3 normal calibration values. This confirms that normal walking/crowd footage is
a meaningful hard-negative source and that camera-specific calibration is required.
The intervals are provisional until frame-by-frame human review is complete; no refit
has been performed yet.

The interval-aware command is:

```powershell
.\.venv\Scripts\python.exe src\calibrate_people_thresholds.py `
  --interval data\shanghaitech\testing\frames\01_0025:1:127 `
  --interval data\shanghaitech\testing\frames\01_0063:1:59 `
  --interval data\shanghaitech\testing\frames\07_0048:1:168 `
  --frame-gap 5 --sample-step 10 --max-pairs-per-video 20 `
  --output results\calibration\hard_negative_check.json
```

---

## 6. Planned experiment matrix

| ID | Model | Data | Scene(s) | Purpose | Status |
|---|---|---|---|---|---|
| E-001 | PatchCore | ShanghaiTech | 1 | first real baseline | Done |
| E-002 | PaDiM | ShanghaiTech | 1 | classical baseline | Planned |
| E-003 | PatchCore | ShanghaiTech | 2, 3 | scene generalization | Planned |
| E-004 | PaDiM | ShanghaiTech | 2, 3 | scene generalization | Planned |
| E-005 | PatchCore | ShanghaiTech | 1 | memory-bank size ablation | Planned |
| E-006 | PatchCore | ShanghaiTech | 1 | coreset ablation | Planned |
| E-007 | PatchCore | ShanghaiTech | 1 | backbone/layer ablation | Planned |
| E-008 | AI-VAD | ShanghaiTech | 1–3 | video-native baseline | Planned |
| E-009 | FUVAS | ShanghaiTech | 1–3 | few-shot curve | Planned |
| E-010 | WinCLIP | ShanghaiTech | 1–3 | zero-shot | Planned |
| E-011 | all | ShanghaiTech | 1–13 | full benchmark | Planned |
| E-012 | best | Avenue | all | cross-dataset | Planned |
| E-013 | all | ShanghaiTech | 1–3 | temporal ablation | Planned |
| E-014 | best | video demo | n/a | Gradio + ONNX FPS | Planned |

---

## 7. Memory and throughput notes

### PatchCore memory model

```
bytes ≈ frames × patches_per_frame × feature_dim × 4
```

For 256×256 and WRN50 layer2+3:

- patches_per_frame ≈ 32 × 32 = 1,024
- feature_dim = 512 + 1,024 = 1,536
- one frame ≈ 6.0 MB
- 700 frames ≈ 4.2 GB
- 2,000 frames ≈ 12 GB (likely unsafe on 7.9 GB RAM)
- 69,976 frames ≈ 440 GB (impossible)

This is why the memory bank is sampled and why 2,000 frames may not be feasible without
further reduction (lower resolution, fewer patches, or a different backbone).

### Test metric memory

Pixel AUROC accumulates one value per pixel per frame:

- 256×256 = 65,536 values per frame
- 11,894 frames ≈ 780 million values
- float32 ≈ 3.1 GB raw, plus sort overhead ≈ 6.2 GB (observed allocation)
- 1,204 frames ≈ 79 million values ≈ 0.3–0.8 GB (worked)

### Throughput observed

| Stage | Rate |
|---|---|
| PatchCore fit (feature extraction + coreset) | ~3 frames/s effective with 47 batches |
| Test inference + metrics, workers=0 | ~3.2 frames/s |
| Full-test inference only, before OOM | ~5.9 frames/s |

Workers were 0. Trying 2–3 workers is a planned optimisation, but RAM is tight.

---

## 8. Reporting requirements

Every final number must be accompanied by:

- model and backbone,
- feature layers,
- coreset ratio,
- memory-bank sampling rule,
- test sampling rule,
- input resolution,
- batch size and workers,
- threshold source,
- package versions,
- whether the visualizer was disabled,
- seed.

Without this, the numbers are not reproducible or comparable.
