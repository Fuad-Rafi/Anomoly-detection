# Frame-Rate Robustness (Phase 0–1)

Status: Phase 0 harness and Phase 1 invariance complete (2026-11-03).

Scope: **10–30 fps only.** 50/60 fps is out of scope because the project has no
genuine high-frame-rate labelled footage. ShanghaiTech testing is distributed as
10 fps frames; Avenue is real 25 fps; 15/20/30 fps runs are emulated by resampling a
real source.

## 1. Definitions

| Term | Meaning |
|---|---|
| `source_fps` | Native rate of the decoded stream (ShanghaiTech 10, Avenue 25) |
| `target_dt` | Physical window between the two AI-VAD frames, in seconds |
| `gap` | `round(source_fps × target_dt)`, clamped to at least 1 |
| `effective_dt` | `gap / source_fps` — the window that was actually used |
| `emulate_fps` | Output cadence used to compare a source at another rate |
| `process_hz` | Decision cadence; defaults to `emulate_fps` |

A detector checkpoint is only valid at the Δt it was fitted on. The scene-1
checkpoint used here was fitted at 24 fps with gap 5, i.e. **0.2083 s**.

## 2. Implementation

| File | Purpose |
|---|---|
| `src/temporal.py` | Gap rounding, effective-Δt reporting, sample plans, folder/video frame sources |
| `src/aivad_core.py` | Batched inference, per-person scoring, selection, tracking |
| `src/eval_aivad.py` | Headless timestamped evaluator with per-pair inference cache |
| `src/evaluate_fps.py` | Alignment by source frame, invariance and event metrics |
| `src/calibrate_from_csv.py` | Percentile thresholds from a normal-footage score CSV |
| `src/report_fps.py` | Markdown summary of the invariance CSVs |
| `src/compare_golden.py` | Guard that the refactor preserves the legacy renderer |
| `src/render_people_video.py` | Thin renderer wrapper; now accepts `--source-fps`, `--target-dt`, seconds-based temporal flags |
| `tests/test_temporal.py`, `tests/test_alignment.py`, `tests/test_candidates.py` | Unit tests (30 passing) |

All temporal parameters in the evaluator are durations (`--smooth-seconds`,
`--cooldown-seconds`, `--focus-timeout-seconds`, `--track-*-seconds`), so they scale
with the decision cadence instead of assuming 10 fps.

### 2.1 Operating point

Thresholds are fixed for every frame-rate run, so differences are attributable to
sampling rather than to re-tuning. They come from a **normal training video**
(`data/shanghaitech/training/videos/01_001.avi`, 24 fps, same Δt), scored once and
reduced to the 90th percentile (`src/calibrate_from_csv.py`):

| Threshold | Value | Note |
|---|---|---|
| Image | 1.22996 | 90th percentile of frame scores |
| Box | 1.2393 | 90th percentile of per-person scores |
| Motion | 0.0 (disabled) | The robust gap-5 checkpoint compresses velocity; its p90 motion gate would silence the fight clip. Recalibrated per Δt in Phase 2. |

The 90th percentile is deliberately permissive: Phase 1 tests **agreement**, not
deployment false-alarm rate. `results/fps/calibration/phase1.json` records the exact
numbers.

### 2.2 Refactor guard

The golden test caught a real bug during extraction: `infer_batch` initially passed
the last frame to AI-VAD's feature extractor, while Anomalib uses the first frame for
appearance/pose features and the last frame only for region detection. A second bug
stored cached candidates in CSV shape (`box`) while `select_persons` expects
`x/y/w/h`; it is now covered by `tests/test_candidates.py`. After the fixes the
refactored renderer reproduces the legacy CSV exactly (49 frames, tolerance 1e-3).
`src/_legacy_render_people_video.py` is the frozen snapshot used by the guard.

## 3. Experiments

| ID | Data | Setup | Purpose |
|---|---|---|---|
| E-FPS-01 | ShanghaiTech `01_0026` | 5/10/20/30 fps, full clip | Invariance at fixed Δt |
| E-FPS-01b | ShanghaiTech `03_0061` | 5 vs 10 fps, full clip | Second-clip check |
| E-FPS-02 | Avenue `01` (25 fps real) | 25 vs 10 fps on 0–30 s | Real cross-fps check |
| E-FPS-03 | ShanghaiTech `01_0026` | 0.05 s at 10 fps, full clip | Sub-frame Δt quantisation |

The 20/30 fps runs process repeated source pairs at the emulated cadence. Inference
is cached per unique `(first, current)` pair, so the GPU cost stays close to the
10 fps run while the smoothing/tracking clock runs at 20/30 Hz.

## 4. Metrics

- Spearman and Pearson correlation of per-frame scores.
- KS statistic and median score ratio.
- Alarm-frame and boxed-frame Jaccard agreement (restricted to common frames).
- Matched alarm events within ±0.5 s.
- First/last alarm offset in **seconds**.
- AUROC, frame F1, and false alarms per minute where ground truth is available.

Acceptance for full-rate runs at fixed Δt: Spearman ≥ 0.95, alarm Jaccard ≥ 0.95,
and delay agreement within one source frame. The 5 fps decimation is expected to be
less stable and is reported separately.

## 5. Results

All full-clip comparisons use `target_dt = 0.2083 s`; the reference is the 10 fps run.
`Alarm J` is the Jaccard overlap of alarm frames on the common source frames; `Box J`
is the same for frames with at least one box. Delay is the first-alarm offset in
seconds.

| Data | Comparison | Frames | Spearman | Pearson | Alarm J | Box J | First-alarm Δ | AUROC ref/run |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `01_0026` | 10 → 5 fps | 252 | **0.983** | 0.992 | **0.938** | 0.231 | −0.40 s | 0.884 / 0.891 |
| `01_0026` | 10 → 20 fps | 503 | **0.999** | **1.000** | **0.994** | **1.000** | 0.00 s | 0.884 / 0.883 |
| `01_0026` | 10 → 30 fps | 503 | **0.999** | **1.000** | **0.987** | **1.000** | 0.00 s | 0.884 / 0.883 |
| `03_0061` | 10 → 5 fps | 120 | **0.996** | 0.997 | **0.991** | 0.839 | 0.00 s | 0.740 / 0.740 |
| Avenue `01` | 25 → 10 fps (real 25 fps) | 299 | **0.987** | 0.994 | 1.000* | 0.765 | 0.00 s | n/a |
| `01_0026` | Δt 0.208 → 0.05 s (10 fps) | 503 | 0.970 | 0.967 | 0.845 | 0.000 | +0.80 s | 0.884 / 0.897 |

\* Avenue's 10 fps run alarms on every common frame at the ShanghaiTech-calibrated
threshold, so the Jaccard is saturated and not informative. The score correlation and
median ratio (1.003) are the meaningful cross-fps results.

The first 20/30 fps pass used a 25–42 s window and showed a −5.1 s first-alarm offset at
the segment boundary because smoothing started with an empty history. The reported rows
are full-clip reruns; that artifact is gone (delay 0.0 s, 3/3 events matched).

### Gate status

| Claim | Result |
|---|---|
| C1: fixed-Δt invariance at full rate | **Pass.** 20/30 fps: Spearman 0.999, alarm J 0.987–0.994, delay 0 s, all 3 events matched. |
| C2: seconds-based temporal logic | **Pass.** Temporal parameters scale with cadence; the alarm timeline is unchanged. |
| 5 fps decimation | **Partial.** Score/alarm agreement holds (0.983/0.938 and 0.996/0.991), but box localization drops because the run can only box every other source frame and boxes are sparse. |
| Cross-dataset Avenue | **Partial.** Ranking is preserved (0.987); alarm Jaccard is saturated by a threshold mismatch. |
| Δt must match the checkpoint | **Confirmed.** `target_dt = 0.05 s` at 10 fps is quantised to **0.1 s**; scores shift (Spearman 0.970) and box overlap collapses. |

### Interpretation

- The frame-rate parameterization is sound: at a fixed Δt, 20/30 fps emulations
  reproduce the 10 fps ranking almost exactly and preserve alarm timing to the frame.
- 5 fps is a decimation stress test, not a supported operating point. It loses half
  the frames, so box Jaccard cannot exceed roughly the frame coverage; score and
  alarm agreement remain strong.
- Avenue provides the only **real** cross-fps signal: a 25 fps source decimated to
  10 fps keeps ρ = 0.987 and box J = 0.765.
- The `0.05 s` run is not evidence that smaller Δt is better. It is evidence that a
  Δt smaller than one frame interval silently becomes `1/fps`, which changes the
  model input and requires a matched refit.

### Artifacts

- `results/fps/invariance_01_0026.csv`
- `results/fps/invariance_03_0061.csv`
- `results/fps/invariance_avenue_01.csv`
- `results/fps/quantisation_01_0026.csv`
- `results/fps/report.md`
- Per-run scores and metadata under `results/fps/<dataset>/<clip>/fps<F>_dt<D>/`
- Logs under `results/fps/logs/`

## 6. Known limits

- ShanghaiTech testing is 10 fps, so the 20/30 fps rows validate the sampler and the
  seconds-based temporal clock, not the value of genuinely higher frame rates.
- Avenue is the only local real cross-fps labelled source (25 fps); no 50/60 fps data.
- The Phase 1 operating point is deliberately permissive (90th percentile of a normal
  training video, motion gate disabled). The reference `01_0026` run has AUROC 0.884
  but F1 0.647 and ~79 false alarms/min; these are not deployment numbers.
- The scene-1 checkpoint is Δt-specific. Comparing 0.04/0.08/0.2 s requires matched
  refits, which is Phase 2.
- The motion gate was disabled because the robust gap-5 checkpoint compresses velocity;
  per-Δt motion calibration is part of Phase 2.

## 7. Real 25 fps Avenue demo

The ShanghaiTech 20/30 fps rows are emulated. To show an actually higher-fps source,
AI-VAD was fitted on Avenue and used to render a native 25 fps clip.

### Setup

- Model: AI-VAD, `persons_only=True`, gap 5 at 25 fps -> **0.2 s**.
- Training: 604 clips from 16 Avenue normal videos, stride 25; 5,053 detections.
- Robust normalization: velocity p1/p99.5 = −1.912 / 180.633; deep 0.503 / 4.482;
  pose 0.022 / 1.940.
- Calibration: `training_videos/01.avi`, first 20 s, 80th percentile.
  Image **1.9651**, box **1.20128**, motion gate disabled.
- Demo clip: `testing_videos/05.avi`, 1,007 frames, 25 fps, GT span 468–785
  (318 positive frames).

### Result

| Metric | Value |
|---|---:|
| Boxed frames | 138 / 1,002 |
| Boxed frames on GT | 128 |
| Box precision | **0.928** |
| Box recall | 0.402 |
| Frame alarms | 195 |
| Alarm precision / recall / F1 | **0.826 / 0.506 / 0.628** |
| Boxed ranges | 599–658, 713–783, 808–819 |

The last range (808–819) is after the labelled event and is a false-positive cluster.
The 80th-percentile operating point was chosen because q90 produced only 38 boxed
frames (all correct, but recall 0.12); q80 trades some precision for a visible demo.

### Artifact

```
results/videos/avenue_05_25fps_annotated.mp4
```

640×360, **25 fps**, 1,002 frames, 40.08 s, boxes + score banner. Scores:
`results/fps/avenue_demo/05/scores.csv`.

### Second clip: 17.avi

`17.avi` is short (426 frames, 25 fps) and almost entirely anomalous in the GT
(379 positive frames, span 0–425). The q80 threshold from clip 05 produced **zero
boxes** here, because clip 17's scores are systematically lower. A per-clip demo
threshold of **image 1.70, box 1.00** was used:

| Metric | Value |
|---|---:|
| Boxed frames | 333 / 421 |
| Boxed frames on GT | 329 |
| Box precision | **0.988** |
| Box recall | **0.868** |
| Frame alarms | 366 |
| Alarm precision / recall / F1 | **0.951 / 0.918 / 0.934** |

Artifact: `results/videos/avenue_17_25fps_annotated.mp4` (640×360, 25 fps,
16.84 s, 421 frames).

This is the clearest demonstration of a real limitation: **a single Avenue threshold
does not transfer across clips**. Clip 05 needed image 1.965 for a usable demo; clip
17 needed 1.70. Camera/event-specific calibration is required, and the per-clip
threshold used here is a demo choice, not a benchmark protocol.

### Blind unseen test: 01.avi

To test the model without tuning on the clip, `01.avi` was run with the fixed q80
operating point (**image 1.9651, box 1.20128**, motion disabled) and the labels were
only inspected afterwards.

| Metric | Value |
|---|---:|
| Clip | 1,439 frames, 25 fps, 57.36 s |
| GT positives | 408 (span 77–1100) |
| Boxed frames | 1,023 / 1,434 |
| Boxed frames on GT | 293 |
| Box precision | **0.286** |
| Box recall | 0.718 |
| Frame alarm precision / recall / F1 | **0.319 / 0.993 / 0.483** |

The blind threshold **over-fires badly** on this clip: boxes appear on normal people
throughout the video. A label-informed sweep found a better trade-off around image
2.20 / box 1.20 (371 boxed frames, 202 on GT, precision 0.54, recall 0.50; frame
precision 0.54, recall 0.77), but even that is weak.

Artifact: `results/videos/avenue_01_25fps_annotated_blind.mp4` (blind q80 result),
scores in `results/fps/avenue_demo/01/scores_blind.csv`.

### Label-informed rerender: 01.avi

For a visually cleaner video, the threshold was tuned to **image 2.20 / box 1.20**.
This uses the labels and is not a blind result:

| Metric | Value |
|---|---:|
| Boxed frames | 366 / 1,434 |
| Boxed frames on GT | 200 |
| Box precision | **0.546** |
| Box recall | **0.490** |
| Frame alarm precision / recall / F1 | **0.540 / 0.770 / 0.635** |

Artifact: `results/videos/avenue_01_25fps_annotated.mp4`, scores in
`results/fps/avenue_demo/01/scores.csv`.

The tuned video is cleaner but still misses about half the anomalous frames. At
around 40 s (inside the GT span) the banner reports a high score but no person box,
so the event is not marked. This is not a success; it is the best trade-off available
with this model on this clip.

### Limits

- Thresholds are calibrated on Avenue **training** footage, so the model has seen
  those videos; this is not a held-out benchmark.
- One clip only; Avenue's 25 fps is real, but 50/60 fps remains out of scope.
- The motion gate is disabled; per-Δt motion calibration belongs to Phase 2.
