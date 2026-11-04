# Video Anomaly Detection — Complete Project Guide

Last updated: 2026-11-03
Owner: project workspace `F:\Developer Stack\ML vision`
Status: environment ready (GPU verified); all three datasets prepared; first real PatchCore
result on ShanghaiTech scene 1 obtained; corrected people-only AI-VAD demo delivered
(98/98 boxed frames on ground truth, per-person precision 0.888). Training runs continue
from here.

**Current focus:** frame-rate-robust detection. Standard CCTV runs 25/30 fps, high-end
50/60 fps. The model is time-based (`Δt ≈ 0.2 s`), so `gap = round(fps × Δt)` runs on any
source today. Phase 1 (2026-11-03) verified invariance at fixed Δt across 10–30 fps:
20/30 fps emulations match the 10 fps run (Spearman 0.999, alarm Jaccard 0.987–0.994,
zero delay); 5 fps retains score/alarm agreement but loses box localization. 50/60 fps is
out of scope (no genuine high-fps labelled data). The open question is whether smaller Δt
(0.04/0.08 s) improves fast-event detection — Phase 2. See `docs/FPS_ROBUSTNESS.md` and
`RESEARCH_ROADMAP.md` §2b.

---

## 1. Plain-English summary

A security camera records the same corridor every day. Almost everything that happens is
normal: people walk, stand, carry bags. Very rarely something weird happens: someone runs,
a fight breaks out, a bicycle rides through a pedestrian area.

We cannot collect thousands of examples of "weird", because weird is rare and unpredictable.
So we teach the computer using **only normal videos**. It learns what normal looks like.
Later, when it sees a video, it scores how unlike-normal each frame and each region is.
High score = "look here, this is unusual".

In this project we:

1. Train several detectors on the same normal footage.
2. Test them on clips that contain both normal and abnormal events.
3. Measure who finds the weird parts best, fastest, and with the fewest false alarms.
4. Add a temporal layer so isolated noisy frames do not trigger alarms.
5. Build a small web app where a user uploads a clip and sees the anomalies marked.

## 2. Objectives

### Primary goals

- Build an end-to-end video anomaly detection pipeline on a single RTX 3060.
- Compare a classical method (PaDiM), a modern image baseline (PatchCore), a video-native
  method (AI-VAD), a recent few-shot method (FUVAS), and a zero-shot method (WinCLIP).
- Report rigorous, comparable metrics on standard benchmarks.
- Demonstrate the difference between **frame-wise** detection and **video-native** detection.
- Deliver a runnable demo and a readable report suitable for a portfolio/interview.

### Success criteria

- Reproducible results table with per-scene and mean metrics.
- At least one ablation per model family.
- A working Gradio demo that accepts a video and produces an annotated output.
- Documented limitations and error analysis (not just a happy-path demo).

### Non-goals

- Real-time multi-camera deployment at production scale.
- Training large vision-language models from scratch.
- Beating published state of the art on every benchmark.
- Commercial use of the datasets.

## 3. Background (minimum theory)

### One-class / unsupervised anomaly detection

Training data contains only normal samples. The model learns a representation of normality
and assigns an anomaly score to new samples. No defect labels are needed at training time.

### Two outputs per frame

1. **Image/frame score** — is this frame abnormal? Used for accept/reject decisions.
2. **Pixel score map** — where in the frame is the anomaly? Used for localization and masks.

### Frame-wise vs video-native

- **Frame-wise:** treat each frame as an independent image, run an image anomaly detector,
  then optionally smooth scores over time. Simple, strong baselines, but blind to motion.
- **Video-native:** use temporal information (optical flow, tracked people, clip features).
  Better for "sudden motion" anomalies (running, chasing, fighting), which are exactly the
  hard cases in ShanghaiTech.

### Threshold-free vs thresholded metrics

- **AUROC / PRO** do not need a threshold: they measure ranking quality.
- **F1** needs a threshold. The threshold must be chosen on a validation split, never on test.

## 4. Scope and decisions (decision log)

| Decision | Choice | Reason |
|---|---|---|
| Domain | Video anomaly detection (surveillance) | User accepted surveillance after industrial video turned out not to have a standard benchmark |
| Framework | **Anomalib 2.7.0** | User chose library-based implementation; fast, reproducible, includes video models |
| Primary dataset | **ShanghaiTech Campus** | 13 scenes, pixel masks, 130 anomalies, standard benchmark |
| Secondary dataset | **CUHK Avenue** | Small, quick, auto-download support |
| Extra dataset | **UCSD Ped1/Ped2** | Small sanity-check benchmark |
| First scope | **3 scenes**, then all 13 | Fast iteration, then full benchmark |
| Models | PaDiM, PatchCore, AI-VAD, FUVAS, WinCLIP | One per family; tells a progression story |
| WinCLIP timing | After baselines work | User choice; avoids blocking on zero-shot setup |
| Demo | **Gradio** | User choice; fastest for video upload and annotated output |
| Python | **3.10 venv** | Anomalib 2.7 supports 3.10–3.12; system default 3.13 is outside tested classifiers |
| GPU | Local RTX 3060 12 GB | No cloud cost |

## 5. Datasets

### 5.1 ShanghaiTech Campus (primary)

- 13 scenes from a university campus, varied lighting and camera angles.
- Training data: normal videos only.
- Testing data: normal and anomalous clips, with frame-level and pixel-level ground truth.
- Commonly cited stats: ~330 training clips, ~107 test clips, 130 abnormal events,
  270,000+ training frames.
- Anomaly types include: running, chasing/fighting, wrong-direction movement, bicycles or
  other vehicles in pedestrian areas, sudden motion.
- License: BSD-2-Clause (research use).
- Official page: `https://svip-lab.github.io/dataset/campus_dataset.html`

Expected layout after preparation:

```
data/shanghaitech/
├── training/
│   ├── videos/            # normal .avi clips
│   ├── frames/            # extracted frames
│   └── converted_videos/  # re-encoded copies used by Anomalib
└── testing/
    ├── frames/            # test clip frames
    ├── test_frame_mask/   # frame-level labels
    └── test_pixel_mask/   # pixel-level masks
```

Anomalib's `ShanghaiTech` datamodule requires `training/` and `testing/` to be immediate
children of the configured root. On first use it re-encodes training videos to a
pyav-readable format; that step can take a while.

### 5.2 CUHK Avenue (secondary)

- 16 training videos (normal), 21 test videos (normal + abnormal).
- 47 abnormal events: throwing objects, loitering, running, wrong direction.
- ~30,652 frames total.
- Anomalib provides an `Avenue` datamodule that can auto-download the dataset and convert
  `.mat` masks to `.png`.
- License: academic research only.
- Official page: `http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html`

Current local layout (already downloaded and extracted):

```
data/avenue/
├── Avenue Dataset/
│   ├── training_videos/   # 16 normal clips
│   ├── testing_videos/    # 21 clips, normal + abnormal
│   ├── training_vol/
│   └── testing_vol/
├── ground_truth_demo/
├── Avenue_Dataset.zip
└── ground_truth_demo.zip
```

Note: Anomalib expects the dataset root to contain `training_videos/` and
`testing_videos/` directly. The current extraction has an extra `Avenue Dataset/`
folder, so those contents must be moved up one level before running Anomalib.

### 5.3 UCSD Ped1 / Ped2 (extra sanity check)

- Ped1: 34 training clips, 36 test clips; some pixel masks.
- Ped2: 16 training clips, 12 test clips; some pixel masks.
- Abnormal events: vehicles (bikes, cars, carts), skaters, wrong-direction pedestrians.
- Videos are stored as folders of frames, not `.avi`.
- License: research use.
- Official page: `http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm`

Current local layout:

```
data/ucsdped/UCSD_Anomaly_Dataset.v1p2/
├── UCSDped1/{Train,Test}
└── UCSDped2/{Train,Test}
```

Test folders contain `*_gt/` subfolders with ground-truth masks.

### 5.4 Download status at time of writing

| Dataset | Status | Location |
|---|---|---|
| Avenue | Done (776 MB zip + 0.9 MB GT, extracted, layout fixed) | `data/avenue/` |
| UCSD Ped1/Ped2 | Done (706 MB tar.gz, extracted) | `data/ucsdped/` |
| ShanghaiTech | Done (~6.5 GB, 7 parts joined, extracted, 330 train videos re-encoded) | `data/shanghaitech/` |

The browser/OneDrive download of `splits.zip` failed (HTTP 403, then the partial file
disappeared). The same 7-part archive was fetched from the Hugging Face mirror
`123zdm/ShanghaiTech`, verified byte-exact against the expected part sizes, joined, and
extracted.

ShanghaiTech distribution is a multi-part archive:

```
shanghaitech.tar.gz.aa  ~1 GB
shanghaitech.tar.gz.ab  ~1 GB
shanghaitech.tar.gz.ac  ~1 GB
shanghaitech.tar.gz.ad  ~1 GB
shanghaitech.tar.gz.ae  ~1 GB
shanghaitech.tar.gz.af  ~1 GB
shanghaitech.tar.gz.ag  ~525 MB
```

Source folder in the OneDrive web UI:
`OpenSources > ano_pred_cvpr_2018 > ShanghaiTechCampus > splits`

The downloaded archive contained these parts; all 7 sizes were verified before joining.

## 6. Methods

| Model | Family | Training data | One-line idea | Why included |
|---|---|---|---|---|
| **PaDiM** | Frame-wise, statistical | Normal frames | Fit a Gaussian per patch position; score by Mahalanobis distance | Classical baseline, cheap |
| **PatchCore** | Frame-wise, memory bank | Normal frames | Store a coreset of normal patch features; score by k-NN distance | Strong modern image baseline |
| **AI-VAD** | Video-native | Normal clips | Combine person-detector appearance features and optical-flow motion features | Detects motion anomalies that frame-wise models miss |
| **FUVAS** | Video-native, few-shot | A few normal clips | Few-shot video anomaly segmentation | Recent method; "latest tech" headline |
| **WinCLIP** | Zero-shot, frame-wise | None | Use CLIP image/text similarity to prompts such as "flawless" vs "damaged" | Shows foundation-model zero-shot capability |

### 6.1 PaDiM (detail)

- Frozen CNN backbone (e.g. WideResNet-50), features from intermediate layers.
- Concatenate and resize features so every spatial position has one vector.
- For each position, fit a multivariate Gaussian over training images, with covariance
  regularization/shrinkage.
- Test score for a patch = Mahalanobis distance to that position's Gaussian.
- Frame score = maximum patch score.

### 6.2 PatchCore (detail)

- Same frozen backbone, patch features from layers 2 and 3, local neighborhood pooling.
- Build a memory bank of normal patch features.
- Subsample a coreset (~1%) to keep inference fast.
- Score each test patch by distance to its nearest normal patch.
- Frame score = maximum patch score, optionally weighted by local neighborhood.

### 6.3 AI-VAD (detail)

- Uses a person detector plus optical-flow features.
- Models appearance and motion normality separately.
- Produces per-region and per-frame anomaly scores.
- Video-native: requires clips, not single frames.
- Because it models motion, it is the natural counterpoint to PaDiM/PatchCore.
- Product mode uses a deployment-matched frame gap, component diagnostics, IoU track
  histories, persistence, hysteresis, and camera-specific thresholds.
- Temporal gaps must be matched in **seconds**, not frame counts: ShanghaiTech training
  videos are 24 fps and test frames are 10 fps. `src/fit_aivad_gap.py` builds exact
  `(t, t + step)` pairs; `src/recalibrate_aivad_norm.py` replaces outlier-dominated
  min-max normalisation with percentiles.
- Display selection uses an anchor + 90 px radius so both fighters are boxed, plus a
  140 px focus lock that rejects a passing person who briefly outscores the event.
- AI-VAD is the prototype baseline, not a universal fight classifier; supervised event
  verification may be added after false-positive calibration and hard-negative mining.

### 6.4 FUVAS (detail)

- Few-shot unsupervised video anomaly segmentation.
- Learns from a small number of normal clips per scene.
- Evaluated as k-shot (k = 1, 2, 4, 8) to produce a few-shot curve.

### 6.5 WinCLIP (detail)

- Zero-shot: no training on the target dataset.
- Uses CLIP image and text encoders.
- Text prompts contrast normal vs abnormal states.
- Multi-scale windowed scoring for localization.
- Per-frame application in this project; temporal smoothing added on top.

## 7. System architecture

```
Raw videos
   |
   v
[Data layer]
  - ShanghaiTech / Avenue / UCSDped datamodules (Anomalib)
  - Frame extraction, clip sampling, ground-truth masks
   |
   v
[Model layer]
  - PaDiM / PatchCore (frame-wise)
  - AI-VAD / FUVAS (video-native)
  - WinCLIP (zero-shot, frame-wise)
   |
   v
[Score layer]
  - Per-frame anomaly score
  - Per-pixel anomaly map
   |
   v
[Temporal layer]
  - Moving average / median over window
  - IoU track histories and per-track score aggregation
  - Persistence, hysteresis, debounce, and cooldown
  - Detection delay and false-alarm counting
   |
   v
[Output layer]
  - Metrics CSV and plots
  - Heatmap overlays, score timelines
  - Annotated video
  - Gradio demo
```

## 8. Repository layout (target)

```
F:\Developer Stack\ML vision\
├── configs/
│   ├── base.yaml
│   ├── padim.yaml
│   ├── patchcore.yaml
│   ├── aivad.yaml
│   ├── fuvas.yaml
│   └── winclip.yaml
├── data/                     # gitignored
│   ├── avenue/
│   ├── ucsdped/
│   ├── shanghaitech/
│   └── splits.zip
├── src/
│   ├── run.py                # build datamodule + model + engine from YAML
│   ├── video.py              # frame extraction, inference, temporal smoothing, events
│   ├── metrics.py            # AUROC, PRO, F1, false alarms/min, latency
│   ├── collect.py            # merge per-run metrics into summary.csv
│   └── plot.py               # figures and heatmap grids
├── app/
│   └── gradio_app.py
├── notebooks/
├── results/
├── tests/
├── docs/
│   └── PROGRESS.md
├── PROJECT_GUIDE.md
├── README.md
└── requirements.txt
```

## 9. Environment setup

### Why Python 3.10

Anomalib 2.7.0 declares `requires_python >=3.10` but its published classifiers only cover
3.10, 3.11, and 3.12. The system default is Python 3.13.9, which is outside that tested
range. A dedicated 3.10 virtual environment avoids dependency surprises.

### Why CUDA wheels

The system Python currently has `torch 2.10.0+cpu`, so `torch.cuda.is_available()` is
`False`. Installing the CUDA build is required to use the RTX 3060.

### Commands

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# Install CUDA PyTorch FIRST, then Anomalib, so Anomalib does not pull a CPU wheel.
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install "anomalib==2.7.0" gradio

# Verify
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.\.venv\Scripts\python.exe -c "import anomalib; print(anomalib.__version__)"
```

If `torch.cuda.is_available()` is `False` after installing Anomalib, Anomalib or
pytorch-lightning downgraded torch. Re-run the CUDA install line and verify again.

### Actual installed environment (verified 2026-11-02)

| Package | Version |
|---|---|
| Python | 3.10.11 (`.venv`) |
| torch | 2.11.0+cu128 (CUDA True, RTX 3060 detected) |
| torchvision | 0.26.0+cu128 |
| anomalib | 2.7.0 |
| gradio | 6.31.0 |
| av (PyAV) | 17.1.0 |
| open-clip-torch | 3.3.0 |
| torchcodec | 0.16.0 |

`torchvision 0.26` removed built-in video decoding and delegates to **TorchCodec**, which
needs FFmpeg **shared** DLLs. This machine had no FFmpeg. Fix applied:

1. Download a portable shared build:
   `https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl-shared.zip`
2. Extract to `tools/ffmpeg/.../bin`.
3. Copy its `*.dll` files into `.venv/Lib/site-packages/torchcodec/` (the loader searches
   the DLL's own directory). After this, `VideoDecoder` works without PATH changes.

The project also installs the video extras explicitly:
`pip install av open-clip-torch torchcodec`.

### requirements.txt

```
# Install torch/torchvision from the CUDA index, not PyPI:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
torch==2.11.0
torchvision==0.26.0
anomalib==2.7.0
gradio==6.31.0
av==17.1.0
torchcodec==0.16.0
open-clip-torch==3.3.0
opencv-python-headless
scikit-learn
scipy
numpy
pandas
matplotlib
tqdm
onnxruntime
```

## 10. Data preparation runbook

### 10.1 Obtain the split archive

The Chrome/OneDrive route proved unreliable (HTTP 403 to automated requests, then the
partial `.crdownload` vanished). The working route was the Hugging Face mirror:

```python
from huggingface_hub import hf_hub_download
for s in ["aa","ab","ac","ad","ae","af","ag"]:
    hf_hub_download(repo_id="123zdm/ShanghaiTech",
                    filename=f"shanghaitech.tar.gz.{s}",
                    repo_type="dataset",
                    local_dir="data/shanghaitech_parts")
```

Expected sizes: `.aa`–`.af` = 1,073,741,824 bytes each; `.ag` = 550,511,474 bytes.
The joined archive must be 6,992,962,418 bytes.

### 10.2 Unzip and join the parts

```powershell
# Unzip the folder archive
Expand-Archive -Path "data\splits.zip" -DestinationPath "data\shanghaitech_parts"

# Move into the parts folder (adjust if unzip creates a nested folder)
Set-Location "data\shanghaitech_parts"

# Join the 7 parts into one archive
cmd /c copy /b shanghaitech.tar.gz.aa+shanghaitech.tar.gz.ab+shanghaitech.tar.gz.ac+shanghaitech.tar.gz.ad+shanghaitech.tar.gz.ae+shanghaitech.tar.gz.af+shanghaitech.tar.gz.ag shanghaitech.tar.gz
```

Verify the joined file is roughly the sum of the parts (~6.5 GB). If a part is short,
re-download that part before joining.

### 10.3 Extract

```powershell
New-Item -ItemType Directory -Force -Path "F:\Developer Stack\ML vision\data\shanghaitech" | Out-Null
tar -xzf "shanghaitech.tar.gz" -C "F:\Developer Stack\ML vision\data\shanghaitech"
```

If the archive extracts into a top-level folder (for example
`data\shanghaitech\shanghaitech\`), move its contents up so that `training\` and
`testing\` are immediate children of `data\shanghaitech\`.

### 10.4 Fix Avenue layout for Anomalib

Anomalib expects `training_videos/` and `testing_videos/` directly under the root.

```powershell
Move-Item "data\avenue\Avenue Dataset\training_videos" "data\avenue\"
Move-Item "data\avenue\Avenue Dataset\testing_videos"  "data\avenue\"
Move-Item "data\avenue\Avenue Dataset\training_vol"    "data\avenue\"
Move-Item "data\avenue\Avenue Dataset\testing_vol"     "data\avenue\"
```

Keep `data\avenue\ground_truth_demo\` where it is (Anomalib's default `gt_dir`).

### 10.5 Verify datasets

```powershell
Get-ChildItem "data\shanghaitech" | Select-Object Name
Get-ChildItem "data\shanghaitech\training" | Select-Object Name
Get-ChildItem "data\shanghaitech\testing" | Select-Object Name
Get-ChildItem "data\avenue" | Select-Object Name
```

Expected:

- `data\shanghaitech\training\{videos,frames,converted_videos}`
- `data\shanghaitech\testing\{frames,test_frame_mask,test_pixel_mask}`
- `data\avenue\{training_videos,testing_videos,training_vol,testing_vol,ground_truth_demo}`

## 11. Experiment runbook

### 11.1 First smoke test (scene 1, PatchCore)

```python
from anomalib.data import ShanghaiTech
from anomalib.models import Patchcore
from anomalib.engine import Engine

datamodule = ShanghaiTech(
    root="data/shanghaitech",
    scene=1,
    clip_length_in_frames=1,   # frame-wise model
)

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.01,
)

engine = Engine(
    max_epochs=1,
    accelerator="gpu",
    devices=1,
    default_root_dir="results/patchcore/scene01",
)

engine.fit(model=model, datamodule=datamodule)
engine.test(model=model, datamodule=datamodule)
```

Notes:

- Exact keyword names can vary by Anomalib version. Verify against the installed 2.7 API
  before the first run.
- `max_epochs=1` is correct for PaDiM/PatchCore: these methods fit feature statistics, not
  gradients.
- For frame-wise models on video data, use `clip_length_in_frames=1`.
- For AI-VAD/FUVAS, use clips (`clip_length_in_frames >= 2`) as required by the model.

### 11.2 Model matrix

| Model | Data module | Clip length | Training | Notes |
|---|---|---|---|---|
| PaDiM | `ShanghaiTech` / `Avenue` | 1 | `engine.fit` (1 epoch) | Frame-wise |
| PatchCore | same | 1 | `engine.fit` (1 epoch) | Frame-wise |
| AI-VAD | same | 2+ | `engine.fit` | Video-native |
| FUVAS | same | 2+ | few-shot fit | k = 1/2/4/8 |
| WinCLIP | same | 1 | none | Zero-shot; evaluate directly |

### 11.3 Scene selection

- Phase A: scenes **1, 2, 3** (fast iteration).
- Phase B: all **13** scenes for the final table.

### 11.4 Result artifacts

Each run should write:

```
results/<model>/scene<NN>/
├── metrics.csv
├── images/            # heatmap overlays
├── config.yaml        # exact configuration snapshot
└── (optional) exported model
```

`src/collect.py` merges these into:

```
results/summary.csv
```

## 12. Metrics and evaluation protocol

### 12.1 Metrics

| Metric | What it measures | Needs threshold |
|---|---|---|
| Frame AUROC | Ranking of abnormal vs normal frames | No |
| Pixel AUROC | Ranking of abnormal vs normal pixels | No |
| PRO / AUPRO | Region-level localization quality | No |
| F1-max | Best balance of precision/recall | Yes (from validation) |
| False alarms per minute | Practical annoyance rate | Yes |
| Detection delay | Frames from anomaly onset to alarm | Yes |
| FPS | Runtime speed | No |

### 12.2 Protocol rules

- Fit on `training/` normal data only.
- Use `testing/` only for final evaluation.
- Choose thresholds on a validation split, never on test.
- Report mean and standard deviation across scenes, not a single lucky scene.
- Keep resize, normalization, and clip sampling identical across models.
- Record the exact config with every result.

### 12.3 Expected ballpark (for sanity checking only)

Exact numbers depend on protocol, preprocessing, and split handling. Use these only to spot
gross bugs:

- PatchCore on image benchmarks typically reaches very high frame/image AUROC.
- PaDiM is usually a few points lower.
- WinCLIP zero-shot is usually lower than trained baselines on the same benchmark.
- AI-VAD/FUVAS are competitive on motion-heavy anomalies.

If a result is wildly off, check: label mapping, normalization, mask alignment, threshold
selection, and whether frames were accidentally shuffled across splits.

## 13. Temporal aggregation design

Frame scores alone are noisy. The temporal layer turns them into events.

Proposed components:

1. **Smoothing**: moving average or median over a window (e.g. 5, 11, 21 frames).
2. **Peak detection**: max over a sliding window.
3. **Event extraction**: consecutive frames above threshold become an event with start/end.
4. **Hysteresis**: higher threshold to start an event, lower threshold to keep it.
5. **Debouncing**: ignore events shorter than N frames.
6. **Reporting**: false alarms per minute, detection delay, event-level precision/recall.

Ablate the window size and the hysteresis margins.

## 14. Ablations

| Axis | Values | Models |
|---|---|---|
| Backbone | ResNet-50, WideResNet-50, ViT/DINOv2 | PaDiM, PatchCore |
| Feature layers | layer2, layer3, layer2+3 | PaDiM, PatchCore |
| Coreset ratio | 0.001, 0.01, 0.1 | PatchCore |
| Covariance regularization | 0.01, 0.1, 0.5 | PaDiM |
| Clip length | 1, 2, 4 | AI-VAD, FUVAS |
| Few-shot k | 1, 2, 4, 8 | FUVAS |
| Prompt templates / scales | variants | WinCLIP |
| Temporal window | 1, 5, 11, 21 | all |
| Input resolution | 224, 256, 320 | all |

## 15. Milestones and progress tracker

| Phase | Work | Status |
|---|---|---|
| 0a | Requirements and scope decisions | Done |
| 0b | Environment audit (GPU, Python, disk) | Done |
| 0c | Project documentation | Done (this file) |
| 0d | Download Avenue | Done |
| 0e | Download UCSD Ped1/Ped2 | Done |
| 0f | Download ShanghaiTech (~6.5 GB) | Done (HF mirror, byte-verified) |
| 0g | Join and extract ShanghaiTech | Done (330 train videos, 107 test clips) |
| 0h | Create Python 3.10 venv + CUDA PyTorch + Anomalib | Done |
| 0i | Verify GPU with PyTorch | Done (CUDA True, RTX 3060) |
| 0j | Fix TorchCodec + FFmpeg shared DLLs | Done |
| 1a | Repo scaffold and config files | Partial (dirs created; code pending) |
| 1b | Metrics and plotting utilities | Pending |
| 1c | Avenue data pipeline smoke test | Done (15,328 train / 15,324 test frames) |
| 2a | PatchCore on scene 1 (real run) | Done (image AUROC 0.785, pixel AUROC 0.989) |
| 2b | PaDiM + PatchCore on scenes 1–3 | Pending |
| 2c | AI-VAD on scenes 1–3 | Partial (scene 1 fitted at gap 1 and gap 5; people-only `01_0026` demo corrected and evaluated per person; scenes 2–3 pending) |
| 3 | All 13 ShanghaiTech scenes + Avenue | Pending |
| 4a | FUVAS few-shot (k curve) | Pending |
| 4b | WinCLIP zero-shot | Pending |
| 5a | Temporal layer + event metrics | Partial (AI-VAD gap, diagnostics, IoU tracks, persistence, hysteresis, focus lock, and pixel-mask per-person evaluation implemented) |
| 5b | Demo output | Partial (two annotated MP4s rendered; Gradio UI pending) |
| 5c | ONNX export + FPS benchmark | Pending |
| 6 | Report, README results table, error analysis | Pending |

### 15.1 First real result (ShanghaiTech scene 1)

Protocol used for the first run:

- Model: PatchCore, WideResNet-50, layers 2+3, coreset ratio 0.01, 9 neighbors, 256 px.
- Memory bank: every 100th normal training frame (~700 frames).
- Test: every 10th test frame (1,204 frames) to fit in 7.9 GB system RAM.
- Visualizer disabled: Anomalib's `ImageVisualizer` expects `image_path`, but video batches
  carry `video_path` (raises `AttributeError`). Metrics are unaffected.

| Metric | Value |
|---|---|
| Image AUROC | 0.7853 |
| Image F1 | 0.6716 |
| Pixel AUROC | 0.9887 |
| Pixel F1 | 0.5948 |

A full-test attempt (11,894 frames) completed inference but crashed while computing AUROC:
TorchMetrics tried to allocate 6.2 GB for sorting ~780M pixel predictions. Future evaluation
must subsample frames, reduce resolution, or stream metrics per video.

## 16. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Python 3.13 incompatibility | Environment failure | Use dedicated 3.10 venv |
| Anomalib pulls CPU torch | GPU unused | Install CUDA torch first; re-verify after Anomalib install; pin versions |
| OneDrive blocks automated download | No ShanghaiTech | Use the Hugging Face mirror (`123zdm/ShanghaiTech`) or JDownloader 2 |
| Only 7.9 GB system RAM | AUROC OOM on full test; slow dataloading | Subsample test frames, reduce resolution, or stream metrics per video |
| TorchCodec cannot find FFmpeg | All video decoding fails | Portable FFmpeg shared build; copy DLLs into `site-packages/torchcodec` |
| Anomalib visualizer vs video batches | Test crashes after inference | Pass `visualizer=False` to image models |
| Corrupt or short archive part | Extraction fails | Verify each part size before joining; re-download bad parts |
| ShanghaiTech video re-encoding takes long | Slow first run | Run once, keep `converted_videos/`; plan time for it |
| Frame-wise models ignore motion | Miss running/chasing anomalies | Include AI-VAD/FUVAS and analyze the difference explicitly |
| Threshold tuned on test | Inflated results | Validation-only thresholding; document the split |
| Different preprocessing per model | Invalid comparison | Shared preprocessing config; snapshot configs with results |
| VRAM limits | OOM on large models | Batch 8–16, 256px first; reduce for FUVAS/CLIP if needed |
| Dataset licenses | Legal misuse | Research-only; note licenses in README |
| Long runs interrupted | Lost work | Resume-safe runner; skip completed (model, scene) pairs |

## 17. Deliverables checklist

- [ ] `results/summary.csv` with all models and scenes.
- [ ] README results table (mean ± std).
- [ ] Heatmap grid figure (input / ground truth / model outputs).
- [ ] Score timeline figure with anomalies marked.
- [ ] Annotated demo video.
- [ ] Gradio app.
- [ ] ONNX export and CPU/GPU FPS table.
- [ ] Report with methods, results, ablations, error analysis, limitations.
- [ ] Reproducible environment (`requirements.txt`, pinned versions, seeds).

## 18. Future plan (roadmap)

### Phase 0 — Foundation (current)

- Finish ShanghaiTech download.
- Join and extract archive.
- Install and verify GPU environment.
- Scaffold the repository.

### Phase 1 — Minimal pipeline

- Shared data loader wrappers and preprocessing.
- Metrics module and unit tests.
- Avenue end-to-end smoke test (small, fast).

### Phase 2 — Baselines

- PatchCore scene 1 smoke test.
- PaDiM and PatchCore on scenes 1–3.
- AI-VAD on scenes 1–3.
- AI-VAD gap experiment and held-out normal calibration.
- Initial scene-2/3 calibration and mask-zero walking/crowd hard-negative screening;
  final gap-matched fitting and hard-negative refit remain pending.
- Second-video validation (`03_0061`) confirmed ordinary walking false positives; current
  AI-VAD should not yet be described as a reliable fight detector.
- Track-level false-alarm and event metrics.
- First heatmaps and timelines.

### Phase 3 — Full benchmark

- All 13 ShanghaiTech scenes.
- Avenue as a second benchmark.
- Per-scene and mean tables.
- Error analysis: object vs motion anomalies, small regions, night scenes.

### Phase 4 — Recent methods

- FUVAS few-shot curve.
- WinCLIP zero-shot with prompt/scale ablation.
- Comparison table: trained vs few-shot vs zero-shot.

### Phase 5 — Video layer and demo

- Temporal smoothing and event extraction.
- False alarms per minute and detection delay.
- Gradio upload -> annotated video.
- ONNX export and FPS benchmark.
- Optional webcam live mode.

### Phase 6 — Report and polish

- Final report with ablations and limitations.
- Clean README, figures, and reproduction instructions.

### Stretch goals

- Add EfficientAd or another recent image model as a sixth row.
- Cross-dataset generalization (train on ShanghaiTech scenes, test on Avenue).
- Active learning loop: label only the most anomalous clips.
- Multi-camera or streaming prototype with OpenCV.

## 19. Troubleshooting

### `torch.cuda.is_available()` is False

- Confirm the CUDA build is installed: `pip show torch`.
- Reinstall: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`.
- Check with `nvidia-smi` that the driver sees the GPU.

### Anomalib import or dependency conflict

- Confirm the venv is active and Python is 3.10.
- Pin `anomalib==2.7.0`.
- If Anomalib downgraded torch, reinstall the CUDA wheel.

### ShanghaiTech download returns 403 or a web page

- The OneDrive share must be opened in a browser or handled by JDownloader 2.
- Download managers cannot enumerate a folder share from the short link.

### `splits.zip` stays as `.crdownload`

- The download is still in progress or paused.
- Check the file size in Chrome; do not rename or move it while downloading.

### Join fails or extraction reports a truncated file

- Verify each part: `.aa`–`.af` ~1 GB, `.ag` ~525 MB.
- Re-download any short part, then rejoin.

### Anomalib cannot find video files

- Ensure `training/` and `testing/` are immediate children of the dataset root.
- Ensure the expected subfolders (`frames/`, `test_frame_mask/`, `test_pixel_mask/`) exist.

### Video decoding errors

- Anomalib requires `av`/pyav; it is installed as a dependency.
- ShanghaiTech training videos are auto re-encoded on first use.

## 20. Glossary

| Term | Meaning |
|---|---|
| One-class | Trained on normal data only |
| Anomaly score | How unlike-normal a frame or patch is |
| Frame AUROC | Ranking quality for frame-level anomaly detection |
| Pixel AUROC | Ranking quality for pixel-level localization |
| PRO / AUPRO | Region-based localization metric |
| Coreset | A subsample that represents a larger set |
| Memory bank | Stored normal feature vectors used for nearest-neighbor scoring |
| Few-shot | Learning from a small number of normal examples |
| Zero-shot | No target-dataset training at all |
| Temporal smoothing | Averaging/maxing scores over time to reduce noise |
| False alarms per minute | How often the system cries wolf |
| Detection delay | How many frames late an alarm is |

## 21. References

### Datasets

- ShanghaiTech Campus: `https://svip-lab.github.io/dataset/campus_dataset.html`
- CUHK Avenue: `http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html`
- UCSD Anomaly Detection: `http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm`

### Software

- Anomalib: `https://github.com/open-edge-platform/anomalib`
- PyTorch CUDA wheels: `https://download.pytorch.org/whl/cu128`
- Gradio: `https://www.gradio.app/`

### Papers (starting points)

- Defard et al., PaDiM (2020)
- Roth et al., PatchCore (2022)
- Liu et al., Future Frame Prediction for Anomaly Detection (CVPR 2018) — ShanghaiTech
- Lu, Shi, Jia, Abnormal Event Detection at 150 FPS in Matlab (ICCV 2013) — Avenue
- Jeong et al., WinCLIP (2023)
- AI-VAD and FUVAS reference papers (see Anomalib documentation)

## 22. Changelog

| Date | Change |
|---|---|
| 2026-11-02 | Project kickoff; scope and model choices decided; environment audited; Avenue and UCSD downloaded and extracted; ShanghaiTech download started; documentation created |
| 2026-11-02 | Environment built (Python 3.10.11, torch 2.11.0+cu128, anomalib 2.7.0); GPU verified; FFmpeg/TorchCodec fixed |
| 2026-11-02 | ShanghaiTech obtained from HF mirror, joined byte-exact, extracted; Avenue layout fixed |
| 2026-11-02 | Data pipelines verified; PatchCore scene 1 real run: image AUROC 0.7853, pixel AUROC 0.9887; full-test AUROC OOM discovered (7.9 GB RAM) |
| 2026-11-02 | Full journey set added: `docs/JOURNEY.md`, `docs/DECISIONS.md`, `docs/CHALLENGES.md`, `docs/EXPERIMENTS.md`; combined PDF built (`docs/Project_Journey.pdf`, 47 pages) |
| 2026-11-02 | Annotated demo video rendered: `results/videos/shanghaitech_scene01_01_0136_annotated.mp4` (529 frames, 247 ANOMALY frames, max score 82.35); renderer `src/render_video.py` |
| 2026-11-02 | AI-VAD fitted on scene 1 (738 clips, 2,262 person detections, 5:02); corrected people-only fight video rendered with density-estimator region scores: `results/videos/shanghaitech_scene01_01_0026_people_annotated.mp4` (161 frames, 91 boxed) |
| 2026-11-03 | Diagnosed gap/fps mismatch (24 fps training, 10 fps test); fitted gap-matched scene-1 checkpoint (2,757 clips, 8,788 detections, 34:30) and added percentile normalisation. Per-person pixel-mask evaluation and focus-lock selection. Final fight video: 138 frames, 98/98 boxed frames on GT, per-person precision 0.888 / recall 0.812. Retracted the gap-5 "clean" render (18.2% precision). |
