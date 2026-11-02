# Video Anomaly Detection (ShanghaiTech / Avenue / UCSD Ped)

Teach a computer to watch surveillance video and flag unusual events by training it on
**normal footage only**, then compare several detectors on the same splits and ship a small
web demo.

> **Plain-English version:** show the computer lots of quiet, normal videos. Later, when
> something strange happens (someone runs, fights, or a bike appears where it should not),
> it should shout "weird!" and mark where.

**Status:** 2026-11-02 — AI-VAD product-hardening pass in progress.
All three datasets are downloaded and prepared. PatchCore on ShanghaiTech scene 1 gives
**image AUROC 0.785 / pixel AUROC 0.989** (subsampled evaluation, see below).

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

- ShanghaiTech scene 1, clip `01_0026` (two people fighting), frames 260-420, 10 fps.
- Video-native **AI-VAD** with `persons_only=True`; red boxes only on people.
- Temporal smoothing (window 15), image threshold 1.3, pixel threshold 1.1.
- 161 frames, **91 with boxes**, first/last boxed 311/405, max smoothed score 2.40.
- Person boxes use AI-VAD's density-estimator region scores, not the maximum map value
  inside each detector box; this removes the earlier pre-fight walker box at frame 263.
- Checkpoint: `results/aivad/scene01/AiVad/ShanghaiTech/v0/weights/lightning/model.ckpt`
  (fitted in 5 min 02 s; 738 clips; 2,262 person detections).
- Scores: `results/videos/shanghaitech_scene01_01_0026_people_annotated.scores.csv`
- Samples: `results/videos/01_0026_people_samples/`

The frame-wise PatchCore model **cannot** do this: on the same fight clip it scored 44-56
with no boxes. AI-VAD uses person detection, optical flow, pose, and CLIP features, which is
why it catches the motion anomaly.

### Product-hardening direction

The prototype now treats AI-VAD as a per-person temporal signal rather than an independent
per-frame alarm. The renderer supports a configurable frame gap, logs velocity/pose/deep
components and per-person scores, associates boxes with lightweight IoU tracks, and applies
track persistence before drawing an alert. Thresholds should be calibrated on held-out normal
walking footage:

```powershell
.\.venv\Scripts\python.exe src\calibrate_people_thresholds.py `
  --frames-dir data\shanghaitech\testing\frames\NORMAL_CLIP `
  --frame-gap 5 --output results\calibration\normal.json
```

The next product validation target is fewer than 1--2 false alerts per camera-day while
retaining high recall on labelled fights, falls, running, and trespassing events. ShanghaiTech
is a research benchmark; deployment requires camera-specific normal footage and review.

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
