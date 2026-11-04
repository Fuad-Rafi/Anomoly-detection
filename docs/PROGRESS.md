# Progress Log

Living document. Update after each work session.

---

## 2026-11-02 — Session 1: kickoff, scope, audit

### Decisions

- Domain: **video anomaly detection** on surveillance datasets.
- Framework: **Anomalib 2.7.0**.
- Primary dataset: **ShanghaiTech Campus**; secondary: **Avenue**, **UCSD Ped1/Ped2**.
- Scope: **3 scenes first**, then all 13.
- Models: PaDiM, PatchCore, AI-VAD, FUVAS, WinCLIP (zero-shot last).
- Demo: **Gradio**.
- Python: dedicated **3.10 venv**.

### Audit findings

- GPU: RTX 3060 12 GB, driver 621.01, CUDA UMD 13.5.
- Default Python 3.13.9; Python 3.10 available.
- System PyTorch was CPU-only (`2.10.0+cpu`).
- No Anomalib, Gradio, timm, lightning, or av initially.
- `tar` available; no 7-Zip, no conda/uv.
- F: had ~69 GB free.

---

## 2026-11-02 — Session 2: build, data, first training result

### Environment (done)

- Created `.venv` with **Python 3.10.11**.
- Installed **torch 2.11.0+cu128**, **torchvision 0.26.0+cu128**.
- Verified **CUDA True, NVIDIA GeForce RTX 3060**.
- Installed **anomalib 2.7.0**, **gradio 6.31.0**, **av 17.1.0**,
  **open-clip-torch 3.3.0**, **torchcodec 0.16.0**.
- Verified all five models import: `Padim`, `Patchcore`, `AiVad`, `Fuvas`, `WinClip`.

### TorchCodec / FFmpeg fix

- torchvision 0.26 delegates video decoding to TorchCodec, which needs FFmpeg **shared** DLLs.
- Downloaded portable shared build
  (`ffmpeg-master-latest-win64-gpl-shared.zip`, FFmpeg 9 / avcodec-63).
- Copied its DLLs into `.venv/Lib/site-packages/torchcodec/`.
- Verified `VideoDecoder` decodes Avenue and ShanghaiTech `.avi` files.

### Data (done)

| Dataset | Result |
|---|---|
| Avenue | 776 MB zip + 0.9 MB GT; extracted; moved `training_videos`, `testing_videos`, `training_vol`, `testing_vol` up to `data/avenue/`; masks converted |
| UCSD Ped1/Ped2 | 706 MB tar.gz; extracted to `data/ucsdped/UCSD_Anomaly_Dataset.v1p2/` |
| ShanghaiTech | Browser/OneDrive route failed (403, partial file vanished); fetched 7 parts from HF mirror `123zdm/ShanghaiTech`; verified sizes; joined to 6,992,962,418 bytes; extracted; moved nested folder up; 330 training videos re-encoded to `training/converted_videos`; 107 test clips with `.npy` masks |

Verified counts:

- ShanghaiTech: **330** training videos, **107** test frame folders, **107** frame masks,
  **107** pixel masks. Scene 1 = **69,976** train clips / **11,894** test clips at stride 1.
- Avenue: **15,328** train frames / **15,324** test samples at clip length 1.

### Smoke tests

- Avenue datamodule loads.
- PatchCore smoke fit (5 batches) runs on GPU.
- `engine.test` crashed in `ImageVisualizer` because video batches carry `video_path`, not
  `image_path`. Fix: `visualizer=False` for image models on video data.
- Some original ShanghaiTech videos decode inconsistently through TorchCodec; the intended
  Anomalib re-encode to XVID fixed the frame-count mismatch.

### First real experiment

- PatchCore, scene 1, WideResNet-50 layers 2+3, coreset 0.01, 9 neighbors, 256 px.
- Memory bank: every 100th normal training frame (~700 frames).
- Test: every 10th test frame (1,204 frames).
- Fit: 239.5 s. Test: 379.9 s.

| Metric | Value |
|---|---|
| image_AUROC | 0.7853 |
| image_F1Score | 0.6716 |
| pixel_AUROC | 0.9887 |
| pixel_F1Score | 0.5948 |

Checkpoint: `results/patchcore/scene01/Patchcore/ShanghaiTech/v0/weights/lightning/model.ckpt`.

### Annotated demo video (E-002)

- Wrote `src/render_video.py`: loads the E-001 checkpoint, runs frame-by-frame inference,
  thresholds the anomaly map, draws red boxes and a score banner.
- Rendered ShanghaiTech scene 1, clip `01_0136` (529 frames).
- Display thresholds after a sweep: pixel 55, image 58, min box area 200, boxes only on
  ANOMALY frames.
- Result: 247 ANOMALY frames, 245 with boxes, first/last boxed frame 169/431, max score 82.35,
  856x480 @ 10 fps, 4.4 MB, render time 19.8 s.
- Artifacts: `results/videos/shanghaitech_scene01_01_0136_annotated.mp4` and
  `...scores.csv`; sample frames in `results/videos/01_0136_samples/`.

### People-only AI-VAD demo (E-003)

- Confirmed PatchCore cannot handle the fight clip `01_0026` (scores 44-56, no boxes).
- Fitted `AiVad(persons_only=True, enable_foreground_detections=False)` on scene 1,
  clip length 2, stride 100: 738 clips, 2,262 person detections, 5 min 02 s.
- Score-only calibration on 504 pairs: normal ~1.04, fight 1.41-1.63; smoothing window 15
  improves frame-level F1 to 0.73.
- Rendered frames 260-420 with smoothing 15, image threshold 1.3, box threshold 1.1:
  161 frames, 91 with boxes, first/last boxed 311/405, max smoothed score 2.40.
- Corrected the renderer to use AI-VAD density-estimator `box_scores` per person instead of
  the anomaly-map maximum inside each box; the earlier walker box at frame 263 disappeared.
- Handled AI-VAD's empty-person crash (`IndexError` in `amax`) as normal.
- Artifacts: `results/videos/shanghaitech_scene01_01_0026_people_annotated.mp4`,
  `...scores.csv`, `results/videos/01_0026_people_samples/`.

## 2026-11-03 — Session 3: fixing the walker boxes

### Diagnosis

- Re-scored both fight renders against the frame mask:
  gap-5 had **6/33 boxed frames on GT (18.2%)**; gap-1 had **86/91 (94.5%)**.
- The gap-5 boxes were mostly at frames 168-200, before the fight (GT 278-397).
- Root causes: training clips used consecutive frames while inference used a gap of 5;
  training videos are 24 fps but test frames are 10 fps, so the real-time gaps differed
  by ~12x; velocity min-max normalisation was dominated by outliers (max 39-70).

### New code

- `src/fit_aivad_gap.py` — exact `(t, t+step)` clip indexer and gap-matched fit.
- `src/recalibrate_aivad_norm.py` — robust percentile normalisation patched into the
  checkpoint.
- `src/evaluate_people_csv.py` — frame precision/recall and candidate distributions.
- `src/tune_people_score.py` — offline weight/gate/threshold sweep.
- `src/analyze_people_pairs.py`, `src/analyze_people_clusters.py`,
  `src/analyze_cluster_tracks.py` — tested interaction heuristics.
- `src/render_people_video.py` — weighted/sum scoring, per-person candidate logging,
  anchor+radius selection, and temporal focus lock.

### Gap-matched fit

- Scene 1, 2,757 clips, step 5 at 24 fps (0.208 s), inference gap 2 at 10 fps (0.2 s).
- 8,788 person detections; 34 min 30 s.
- Robust normalisation from 496 regions: velocity max 43.24 (p99.5), deep 4.64,
  pose 1.68.

### Per-person evaluation

- Pixel masks used to label 224 positive and 1,165 negative candidates in `01_0026`.
- AUC: weighted score 0.852, pose 0.827, velocity 0.796, deep 0.594.
- Score-only logistic regression reached 0.893 AUC on a temporal holdout, but a fixed
  threshold still produced false boxes; temporal continuity was the missing cue.

### Final demo

- Focus lock: radius 140, ratio 0.8, cluster radius 90, relative 0.5, image 1.4,
  box 1.0, smoothing 15, frames 260-397.
- 138 frames, 856x480 @ 10 fps, 0.91 MB.
- 98/98 boxed frames on GT (100% frame precision).
- Per-person precision 0.888, recall 0.812, F1 0.848 (205 selected boxes).
- First/last box 296/397; GT span 278-397; delay 18 frames.
- Artifact: `results/videos/shanghaitech_scene01_01_0026_people_annotated.mp4`.
- Metrics: `results/aivad/scene01_people_demo_metrics.json`.

### Problems found

- **RAM is only 7.9 GB (5.8 GB free).** Full test (11,894 frames) ran inference but AUROC
  crashed allocating 6.2 GB to sort ~780M pixel predictions.
- **PatchCore memory:** 69,976 frames x ~1,024 patches x 1,536 dims is far too large
  (~440 GB). Normal-frame subsampling is mandatory and must be reported.
- Full test inference is slow (~5.9 fps) with `num_workers=0`; consider 2-3 workers.

### Documentation

- `README.md`, `PROJECT_GUIDE.md`, `docs/PROGRESS.md` updated to match the actual state.
- Added `docs/JOURNEY.md` (complete narrative), `docs/DECISIONS.md` (ADR log),
  `docs/CHALLENGES.md` (problem catalog), `docs/EXPERIMENTS.md` (experiments and tuning).
- Added `docs/build_pdf.py` and generated `docs/Project_Journey.pdf` (47 pages, ~1.1 MB).

---

## 2026-11-03 — Session 4: frame-rate robustness (Phase 0/1)

### Harness

- `src/temporal.py` — time-based sample planning, `gap = round(fps × Δt)`, effective-Δt
  reporting, folder/video frame sources.
- `src/aivad_core.py` — shared batched inference, scoring, selection, tracking.
- `src/eval_aivad.py` — headless timestamped evaluator with per-unique-pair inference
  cache, so 20/30 fps emulation costs the same GPU time as 10 fps.
- `src/evaluate_fps.py` — source-frame alignment, Spearman/Pearson/KS, alarm and box
  Jaccard, event matching, delay in seconds.
- `src/calibrate_from_csv.py`, `src/report_fps.py`, `src/compare_golden.py`.
- `tests/test_temporal.py`, `tests/test_alignment.py`, `tests/test_candidates.py` —
  **30 tests pass**.
- The renderer accepts `--source-fps`, `--target-dt`, and seconds-based temporal flags;
  legacy frame-count behavior is preserved.

### Bugs found by the golden guard

1. The refactor passed the **last** frame to AI-VAD's feature extractor; Anomalib uses
   the **first** frame for appearance/pose features and the last only for regions.
2. Cached candidates were stored in CSV shape (`box`) but `select_persons` expects
   `x/y/w/h`; the fix is covered by a regression test.

### Calibration

- Normal training video `01_001.avi`, 24 fps, same Δt, 90th percentile:
  image 1.22996, box 1.2393, motion disabled.
- Motion is disabled because the robust gap-5 checkpoint compresses velocity; per-Δt
  motion calibration is Phase 2.

### Invariance results (Δt = 0.2083 s)

| Data | Comparison | Spearman | Alarm J | Box J | Delay |
|---|---|---:|---:|---:|---:|
| `01_0026` | 10 → 5 fps | 0.983 | 0.938 | 0.231 | −0.4 s |
| `01_0026` | 10 → 20 fps | 0.999 | 0.994 | 1.000 | 0.0 s |
| `01_0026` | 10 → 30 fps | 0.999 | 0.987 | 1.000 | 0.0 s |
| `03_0061` | 10 → 5 fps | 0.996 | 0.991 | 0.839 | 0.0 s |
| Avenue `01` | 25 → 10 fps (real 25 fps) | 0.987 | 1.000* | 0.765 | 0.0 s |
| `01_0026` | Δt 0.208 → 0.05 s | 0.970 | 0.845 | 0.000 | +0.8 s |

\* Saturated: all common Avenue frames alarm at the ShanghaiTech threshold.

- **Pass:** full-rate invariance at fixed Δt (20/30 fps), including alarm timing and all
  three events.
- **Partial:** 5 fps keeps score/alarm agreement but loses box localization because it
  sees half the frames.
- `0.05 s` at 10 fps quantises to `0.1 s`; this is a Δt mismatch, not a high-fps win.
- Artifacts: `docs/FPS_ROBUSTNESS.md`, `results/fps/report.md`, `results/fps/*.csv`,
  per-run scores under `results/fps/<dataset>/<clip>/`.

### Next

Phase 2: Avenue 25 fps matched fits at gaps 1/2/5 (0.04/0.08/0.2 s), recalibrate
percentiles and the motion gate per Δt, evaluate event recall/delay.

---

## 2026-11-04 — Session 5: real 25 fps Avenue demo

### Code

- `src/fit_aivad_gap.py` now supports `--dataset avenue` via an exact-gap
  `AvenueClipsIndexer`, writes `fit_metadata.json`, and reports the true clip count.
- `src/recalibrate_aivad_norm.py` generalized to Avenue.
- `src/eval_aivad.py` gained `--render-output` / `--render-fps`; it draws boxes and a
  score banner while writing the existing timestamped CSV.
- `src/convert_avenue_labels.py` converts Avenue `*_label.mat` to frame-level `.npy`.
- `tools/run_avenue_demo.ps1` runs fit → recalibrate → calibrate → render → evaluate.

### Fit

- 604 clips from 16 Avenue normal videos, gap 5 at 25 fps (**0.2 s**), stride 25.
- 5,053 person detections; fit time ~5 min.
- Robust normalization: velocity p1/p99.5 = −1.912 / 180.633, deep 0.503 / 4.482,
  pose 0.022 / 1.940.

### Demo

- Clip `05.avi`, native 25 fps, 1,007 frames, GT span 468–785 (318 positives).
- Calibration on `training_videos/01.avi` (0–20 s), 80th percentile:
  image 1.9651, box 1.20128, motion disabled.
- **138 boxed frames**, 128 on GT: box precision **0.928**, recall 0.402.
- Frame alarms: precision **0.826**, recall 0.506, F1 0.628.
- q90 was too sparse (38 boxed frames, all correct, recall 0.12), so q80 was used for a
  visible demo.
- Artifact: `results/videos/avenue_05_25fps_annotated.mp4` (640×360, 25 fps, 40.08 s).
- Scores: `results/fps/avenue_demo/05/scores.csv`.

### Second clip: 17.avi

- 426 frames, 25 fps, GT positives 379 (span 0–425).
- The q80 threshold from clip 05 produced **zero boxes**; a per-clip threshold
  (image 1.70, box 1.00) was needed.
- 333 boxed frames, 329 on GT: box precision **0.988**, recall 0.868.
- Frame alarms: precision **0.951**, recall 0.918, F1 0.934.
- Artifact: `results/videos/avenue_17_25fps_annotated.mp4` (640×360, 25 fps, 16.84 s).
- Scores: `results/fps/avenue_demo/17/scores.csv`.
- Lesson: a single Avenue threshold does not transfer across clips; per-clip/camera
  calibration is required.

### Blind unseen test: 01.avi

- Fixed q80 threshold (image 1.9651, box 1.20128), no tuning on this clip.
- 1,023 boxed frames / 1,434, only 293 on GT: box precision **0.286**, recall 0.718.
- Frame alarm precision 0.319, recall 0.993, F1 0.483.
- The blind video over-fires on normal people.
- Blind artifact: `results/videos/avenue_01_25fps_annotated_blind.mp4`;
  scores `results/fps/avenue_demo/01/scores_blind.csv`.
- Label-informed rerender at image 2.20 / box 1.20: 366 boxed frames, 200 on GT,
  box precision 0.546, recall 0.490; frame precision 0.540, recall 0.770, F1 0.635.
- Tuned artifact: `results/videos/avenue_01_25fps_annotated.mp4`;
  scores `results/fps/avenue_demo/01/scores.csv`.
- Even the tuned version misses about half the anomalous frames.
- Lesson: report blind first; the model is not calibrated for a new clip.

### Next

Use the Avenue fit pipeline for the Phase 2 Δt sweep (gaps 1/2/5), recalibrate the
motion gate per Δt, and compare event recall/delay.

---

## Current state

- Environment: ready.
- Data: ready (all three datasets).
- Code: PatchCore and AI-VAD renderers, gap-matched fitting, robust normalisation,
  pixel-mask per-person evaluation, focus lock, threshold tooling, a timestamped
  fps-aware AI-VAD evaluator, and 30 tests are implemented. The general runner, Gradio
  app, and full benchmark harness remain pending.
- Results: one PatchCore scene-1 benchmark result, a corrected AI-VAD scene-1 demo
  (98/98 boxed frames on GT, 0.888 per-person precision), Phase 1 frame-rate invariance
  (20/30 fps match 10 fps at fixed Δt; 5 fps loses box localization), and real 25 fps
  Avenue demos. Clip 05: box precision 0.928 / recall 0.402. Clip 17: 0.988 / 0.868
  with a per-clip threshold. Blind clip 01: precision 0.286 / recall 0.718; tuned
  2.20/1.20: 0.546 / 0.490.
- Next: Phase 2 Avenue Δt sweep and cross-camera calibration, then PaDiM scene 1 and
  the 3-scene comparison.
- **Current focus:** generalisation and calibration. Fixed-Δt frame-rate invariance is
  solved; the open questions are smaller Δt (0.04/0.08 s) for fast events, and how to
  make thresholds transfer across unseen clips/cameras. Meta-learning is a candidate for
  few-shot calibration, not zero-shot. See `RESEARCH_ROADMAP.md` §2b,
  `docs/FPS_ROBUSTNESS.md`, and `docs/CHALLENGES.md` C-37/C-38.

### Product-hardening status

- AI-VAD supports configurable temporal frame gaps; `src/fit_aivad_gap.py` builds exact
  `(t, t + step)` pairs and `src/recalibrate_aivad_norm.py` applies percentile
  normalisation.
- The renderer logs image, velocity, pose, deep, per-region, candidate, track-match, and
  active-track values; it supports weighted or sum scoring, anchor+radius selection, and
  a temporal focus lock.
- The earlier gap-5 "clean" render is **retracted**: it had 6/33 boxed frames on GT
  (18.2%) because the checkpoint was fitted on gap-1 pairs. The gap/fps mismatch is
  documented as C-29.
- The final gap-1 focus-lock demo has 98/98 boxed frames on GT and per-person precision
  0.888 / recall 0.812. This is a display result tuned on one clip, not a benchmark.
- A second validation render, `03_0061`, still produces walking false positives with the
  scene-1 model (131 boxed frames, mainly 76-174). Per-scene fitting or a supervised
  verifier remains the honest next step for cross-camera use.
- Held-out calibration and 99.5th-percentile thresholds from scene-2/3 normal footage
  remain valid diagnostics, but camera-specific fitting is required before deployment.

## Immediate next actions

0. [ ] Phase 2: Avenue matched fits at Δt 0.04/0.08/0.2 s (gaps 1/2/5 at 25 fps),
   recalibrate the motion gate per Δt, and evaluate event recall/delay.
0b. [ ] Per-camera calibration study: support-only threshold/score normalisation,
    then a meta-calibration or Prototypical/MAML head across scenes (few-shot, not
    zero-shot). Compare against FUVAS k = 1/2/4/8.
1. [x] Create `requirements.txt` from the verified environment.
2. [ ] Implement `src/run.py` (config-driven datamodule + model + engine).
3. [x] Implement frame subsampling for the PatchCore memory bank (runner/config support
   still pending).
4. [ ] Implement memory-safe evaluation (image AUROC full test, pixel AUROC on a sample,
   or streaming per-video metrics).
5. [x] Save/load checkpoints so evaluation does not require refitting (PatchCore and
   AI-VAD checkpoints are saved).
6. [ ] Run PaDiM on scene 1 with the same protocol.
7. [~] Build `results/summary.csv` (PatchCore scene 1 done; more rows pending).
8. [~] Run AI-VAD on scenes 1-3 (scene 1 gap-1 and gap-5 fits done; scenes 2-3 pending).
9. [x] Add temporal smoothing and per-person event metrics (event-level benchmarking still
   pending).
10. [ ] FUVAS few-shot, then WinCLIP zero-shot.
11. [ ] Gradio demo, ONNX export, FPS table.
12. [ ] Final report and README results table.
13. [ ] Fit per-scene AI-VAD models and validate `03_0061`; add a supervised verifier if
    walking false positives persist.

## Open questions

- Best memory-safe evaluation strategy: test-frame stride vs per-video streaming AUROC.
- Whether to keep full `samples` for image-level AUROC and subsample only for pixel AUROC.
- Optimal memory-bank size per scene (400 / 700 / 1,000 frames) for the AUROC/memory tradeoff.
- Whether AI-VAD/FUVAS need separate clip lengths and datamodule settings beyond defaults.
- Whether the UCSD `UCSDped` datamodule accepts our extracted folder layout directly.
- How to normalise anomaly scores and thresholds across unseen cameras/clips without
  label leakage (C-37/C-38).
- Whether meta-learning (Prototypical/MAML) for few-shot normal adaptation beats simple
  per-camera percentile calibration, and whether it is worth the task-design complexity.
- Whether smaller Δt at genuinely high frame rates improves fast-event detection; this
  needs real high-fps labelled data, not upsampled 10 fps frames.

## Risks being watched

- System RAM (7.9 GB) limiting full pixel-level evaluation.
- DataLoader speed with `num_workers=0`.
- Anomalib visualizer incompatibility with video batches (use `visualizer=False`).
- FUVAS/WinCLIP VRAM and model-weight downloads on a 12 GB card.
- Keeping the report honest about subsampled training/evaluation protocols.
