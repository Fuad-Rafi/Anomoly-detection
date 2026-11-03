# Video Anomaly Detection (ShanghaiTech / Avenue / UCSD Ped)

Teach a computer to watch surveillance video and flag unusual events by training it on
**normal footage only**, then compare several detectors on the same splits and ship a small
web demo.

> **Plain-English version:** show the computer lots of quiet, normal videos. Later, when
> something strange happens (someone runs, fights, or a bike appears where it should not),
> it should shout "weird!" and mark where.

**Status:** 2026-11-03 — corrected people-only AI-VAD demo delivered.
All three datasets are downloaded and prepared. PatchCore on ShanghaiTech scene 1 gives
**image AUROC 0.785 / pixel AUROC 0.989** (subsampled evaluation, see below). The fight
demo now has **98/98 boxed frames on ground truth** and **0.888 per-person precision**
(pixel-mask evaluation).

---

## Deliverables

- Five detectors on comparable splits:
  - **PaDiM** — frame-wise statistical baseline
  - **PatchCore** — frame-wise memory-bank baseline (first result available)
  - **AI-VAD** — video-native (appearance + motion)
  - **FUVAS** — few-shot video anomaly segmentation (recent method)
  - **WinCLIP** — zero-shot, no training (added last)
- Metrics: frame-level AUROC, pixel-level AUROC, PRO/AUPRO, F1, false alarms per minute,
  detection delay, FPS.
- Temporal smoothing + event-level alarms.
- Gradio app: upload a video, get an annotated output video and a score timeline.
- Reproducible repo, results table, and a short report.

## Documentation

| File | Purpose |
|---|---|
| `README.md` | This file: overview, status, quick start, first results |
| `PROJECT_GUIDE.md` | Full reference: scope, datasets, methods, architecture, runbook, metrics, risks, roadmap |
| `docs/JOURNEY.md` | Complete narrative: how the project evolved, every step and result |
| `docs/DECISIONS.md` | Decision records (ADR style) with alternatives and consequences |
| `docs/CHALLENGES.md` | Problem catalog: symptom, root cause, fix, status, lesson |
| `docs/EXPERIMENTS.md` | Experiment log, hyperparameters, memory analysis, tuning plan |
| `docs/PROGRESS.md` | Living progress log and next actions |
| `docs/Project_Journey.pdf` | PDF build of the journey, decisions, challenges, and experiments |

## First results

ShanghaiTech scene 1, PatchCore, WideResNet-50 (layers 2+3), coreset 1%, 256 px:

| Metric | Value | Protocol |
|---|---|---|
| Image AUROC | 0.7853 | memory bank: every 100th normal train frame (~700) |
| Image F1 | 0.6716 | test: every 10th test frame (1,204) |
| Pixel AUROC | 0.9887 | threshold from validation split |
| Pixel F1 | 0.5948 | visualizer disabled |

These are **subsampled** numbers, not the final benchmark: the machine has only 7.9 GB RAM,
and a full 11,894-frame test ran inference but ran out of memory while computing pixel AUROC.
A streaming/per-video evaluation is the next engineering step.

## Demo video

An annotated MP4 is available:

```
results/videos/shanghaitech_scene01_01_0136_annotated.mp4
```

- ShanghaiTech scene 1, clip `01_0136` (529 frames, 856×480, 10 fps).
- Red boxes around anomalous regions, with a banner: `ANOMALY`/`NORMAL`, score, box count.
- Display thresholds: pixel 55, image 58; boxes only when the frame is anomalous.
- 247 ANOMALY frames, 245 with boxes; max score 82.35 (truck in the pedestrian area).
- Per-frame scores: `results/videos/shanghaitech_scene01_01_0136_annotated.scores.csv`.
- Sample frames: `results/videos/01_0136_samples/` (normal, first box, peak truck).

Render it yourself:

```powershell
.\.venv\Scripts\python.exe src\render_video.py --scene 01 --clip 01_0136 --threshold 55 --image-threshold 58 --min-area 200
```

### People-only demo (AI-VAD)

```
results/videos/shanghaitech_scene01_01_0026_people_annotated.mp4
```

- ShanghaiTech scene 1, clip `01_0026` (two people fighting), frames 260-397, 10 fps.
- Video-native **AI-VAD** with `persons_only=True`; red boxes only on people.
- 138 frames, 856x480, 0.91 MB.
- **98 boxed frames, all 98 on ground-truth anomaly frames** (100% frame precision).
- Per-person pixel-mask evaluation: **precision 0.888, recall 0.812, F1 0.848**
  (205 selected boxes, 182 true positives, 23 false positives).
- Box ranges 296-322, 325-364, 366-374, 376-397; first/last 296/397
  (GT 278-397, so 18-frame delay); max boxes 3; max smoothed score 2.13.
- Settings: gap 1, smoothing 15, sum scoring, image 1.4, box 1.0, relative 0.5,
  cluster radius 90, focus radius 140, focus ratio 0.8.
- Checkpoint: `results/aivad/scene01_stride25/.../model.ckpt`
  (scene 1, 2,840 clips, 8,788 person detections).
- Scores: `results/videos/shanghaitech_scene01_01_0026_people_annotated.scores.csv`
- Per-person metrics: `results/aivad/scene01_people_demo_metrics.json`
- Samples: `results/videos/01_0026_people_samples/`

The fix for the earlier walker boxes was threefold:

1. **Gap/fps matching.** Training clips used consecutive frames, but the old demo inferred
   at a 5-frame gap. Training videos are 24 fps and test frames are 10 fps, so a "5-frame
   gap" meant 0.21 s in training and 0.5 s at inference. `src/fit_aivad_gap.py` fits exact
   `(t, t + step)` pairs; `src/recalibrate_aivad_norm.py` replaces outlier-dominated
   min-max normalisation with percentiles.
2. **Pixel-mask evaluation.** Person boxes are labelled by overlap with the dataset's
   pixel masks, so per-person precision is measured instead of hidden by frame-level
   scores.
3. **Focus lock.** Once an event is tracked, a new high-scoring person far away is
   rejected; nearby people within 90 px are boxed so both fighters are marked.

> The older `shanghaitech_scene01_01_0026_people_gap5_annotated.mp4` result is **retracted**:
> it had only 6 of 33 boxed frames on ground truth (18.2%). It used the gap-5 inference
> setting with a checkpoint fitted on gap-1 pairs. Do not cite it as an improvement.

The frame-wise PatchCore model **cannot** do this: on the same fight clip it scored 44-56
with no boxes. AI-VAD uses person detection, optical flow, pose, and CLIP features, which is
why it catches the motion anomaly.

### Product-hardening direction

The prototype now treats AI-VAD as a per-person temporal signal rather than an independent
per-frame alarm. The renderer supports a configurable frame gap, score-mode weighting,
per-person component logging, anchor+radius selection, focus lock, and IoU track
persistence. Held-out calibration remains available:

```powershell
.\.venv\Scripts\python.exe src\calibrate_people_thresholds.py `
  --frames-dir data\shanghaitech\testing\frames\NORMAL_CLIP `
  --frame-gap 2 --output results\calibration\normal.json
```

The next product validation target is fewer than 1--2 false alerts per camera-day while
retaining high recall on labelled fights, falls, running, and trespassing events. ShanghaiTech
is a research benchmark; deployment requires camera-specific normal footage and review.

The current demo is tuned on one clip and cropped to the labelled event. It is not a
cross-camera fight detector. A second validation clip, `03_0061`, still produces walking
false positives with the scene-1 model; a per-scene fit or supervised verifier is the
honest next step.

## Environment (installed and verified)

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 12 GB, driver 621.01 |
| Python | 3.10.11 in `.venv` |
| torch | 2.11.0+cu128 (`torch.cuda.is_available()` = True) |
| torchvision | 0.26.0+cu128 |
| anomalib | 2.7.0 |
| gradio | 6.31.0 |
| av / torchcodec / open-clip | 17.1.0 / 0.16.0 / 3.3.0 |
| System RAM | 7.9 GB (a real constraint for full-test pixel metrics) |
| Disk | F: with ~50 GB free after datasets and env |

Video decoding requires **FFmpeg shared DLLs** (TorchCodec). A portable build was copied into
`.venv/Lib/site-packages/torchcodec/`; see `PROJECT_GUIDE.md` section 9.

## Data (all ready)

| Dataset | Status |
|---|---|
| ShanghaiTech Campus | 330 normal training videos, 107 test clips with frame + pixel masks; re-encoded |
| CUHK Avenue | 16 train / 21 test videos, ground truth; layout fixed for Anomalib |
| UCSD Ped1/Ped2 | Train/Test frame folders with masks |

ShanghaiTech was obtained from the Hugging Face mirror `123zdm/ShanghaiTech` after the
OneDrive link blocked automated downloads; all 7 archive parts were size-verified and joined
byte-exact (6,992,962,418 bytes).

## Quick start

```powershell
# Environment already exists; activate or call the interpreter directly
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Data pipelines
.\.venv\Scripts\python.exe -m src.run --config configs/patchcore.yaml --scene 1

# Demo (after the app exists)
.\.venv\Scripts\python.exe app\gradio_app.py
```

## Known issues

- **RAM:** 7.9 GB total. Full-test pixel AUROC OOMs. Use subsampled test frames or a
  streaming metric for the final benchmark.
- **Visualizer:** Anomalib's `ImageVisualizer` expects `image_path`; video batches provide
  `video_path`. Pass `visualizer=False` to image models on video data.
- **Frame sampling:** PatchCore cannot hold 70k frames per scene in memory. The normal-frame
  memory bank is subsampled (e.g. every 100th frame); the report must state this.
- **Temporal gaps:** AI-VAD's velocity features depend on the time between the two input
  frames. Training videos are 24 fps and ShanghaiTech test frames are 10 fps, so frame
  counts are not comparable. Match gaps in seconds (`src/fit_aivad_gap.py`).
- **Demo scope:** the fight video is tuned on `01_0026` and cropped to the labelled event.
  A scene-1 model still produces walking false positives on `03_0061`; per-scene fitting
  or a supervised verifier is the next step.

## Dataset licenses

- **ShanghaiTech Campus** — BSD-2-Clause (research use).
- **CUHK Avenue** — academic research only.
- **UCSD Ped** — research use; see the original dataset page.

Research/portfolio project; do not use the datasets commercially.

## Project layout

```
F:\Developer Stack\ML vision\
├── configs/            # YAML per model (to be filled)
├── data/               # datasets (avenue, ucsdped, shanghaitech, shanghaitech_parts)
├── src/                # data, models, metrics, runner (to be filled)
├── app/                # Gradio demo (to be filled)
├── notebooks/          # exploration and figures
├── results/            # metrics, checkpoints, heatmaps, videos
├── tests/              # sanity checks
├── tools/ffmpeg/       # portable FFmpeg shared build
├── docs/PROGRESS.md
├── PROJECT_GUIDE.md
└── README.md
```
