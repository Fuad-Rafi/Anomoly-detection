# Complete Project Journey — Video Anomaly Detection

**Project:** Unsupervised video anomaly detection on surveillance benchmarks
**Workspace:** `F:\Developer Stack\ML vision`
**Date:** 2026-11-02
**Hardware:** NVIDIA RTX 3060 12 GB, 7.9 GB system RAM, Windows
**Stack:** Python 3.10.11, PyTorch 2.11.0+cu128, Anomalib 2.7.0, Gradio 6.31.0

This document records the full journey: how the idea changed, every major decision,
the environment built, the datasets acquired, the failures and workarounds, the first
experiments, hyperparameters used, and what still needs tuning.

Companion documents:

- `docs/DECISIONS.md` — decision records (ADR style)
- `docs/CHALLENGES.md` — problem catalog with root causes and fixes
- `docs/EXPERIMENTS.md` — experiment log, hyperparameters, tuning plan
- `PROJECT_GUIDE.md` — reference manual and roadmap
- `docs/PROGRESS.md` — short running status log

---

## 0. Executive summary

The project started as an industrial image anomaly-detection idea (PaDiM / PatchCore /
WinCLIP on MVTec-AD), then changed to **video anomaly detection** because the user wanted
video and MVTec-AD contains only still images. The final scope is surveillance video
anomaly detection on **ShanghaiTech Campus**, with **CUHK Avenue** and **UCSD Ped1/Ped2**
as secondary datasets.

The environment was built from scratch: a Python 3.10 venv, CUDA PyTorch, Anomalib 2.7.0,
Gradio, PyAV, OpenCLIP, and TorchCodec. A non-obvious FFmpeg shared-library fix was required
before any video could be decoded. All three datasets were acquired and prepared; ShanghaiTech
was ultimately fetched from a Hugging Face mirror after the official OneDrive route blocked
automated downloads.

One real experiment now exists: **PatchCore on ShanghaiTech scene 1**, with
**image AUROC 0.7853**, **image F1 0.6716**, **pixel AUROC 0.9887**, and **pixel F1 0.5948**
under a deliberately subsampled protocol. The subsampling is required because the machine
has only 7.9 GB of RAM: a full test ran inference but ran out of memory while computing
pixel AUROC.

The project is therefore at the transition from "foundation ready" to "systematic
benchmarking and tuning".

---

## 1. How the idea evolved

### 1.1 Original ask

The initial request was a computer-vision project that:

- uses current technology,
- is not too hard,
- demonstrates a strong ML/vision foundation,
- is suitable for a portfolio or interview.

Several ideas were proposed: text-promptable segmentation with Grounding DINO + SAM 2,
real-time detection with YOLO + tracking, monocular depth with Depth Anything V2,
image retrieval with CLIP, and **industrial anomaly detection**.

### 1.2 First choice: industrial image anomaly detection

The industrial anomaly-detection idea was selected and expanded:

- Datasets: **MVTec-AD**, 15 categories.
- Methods: **PaDiM** (classical), **PatchCore** (modern), **WinCLIP** (zero-shot).
- Metrics: image/pixel AUROC, PRO, F1, runtime.
- Library: **Anomalib**.
- Scope: 3 categories first, then all 15.
- Demo: **Gradio**.
- Ordering: WinCLIP last.

### 1.3 The pivot to video

The user then said: **"i want video"**.

Verification against the official sources and the installed Anomalib source showed:

- **MVTec-AD has no video** — it is a still-image dataset.
- **MVTec AD 2** (the newer benchmark) is also still images: 8,000+ high-resolution images,
  not clips.
- Anomalib 2.7 supports video datamodules: `Avenue`, `ShanghaiTech`, `UCSDped`.
- Anomalib 2.7 also contains video-native models: `AiVad`, `Fuvas`.

The user accepted surveillance video (as opposed to insisting on an industrial video
benchmark). The project became:

> Train detectors on normal surveillance footage only, then detect and localize abnormal
> events in test video, comparing frame-wise, video-native, few-shot, and zero-shot methods.

### 1.4 Final project definition

| Element | Final choice |
|---|---|
| Task | Unsupervised video anomaly detection |
| Primary dataset | ShanghaiTech Campus (13 scenes) |
| Secondary datasets | CUHK Avenue, UCSD Ped1/Ped2 |
| Framework | Anomalib 2.7.0 |
| Models | PaDiM, PatchCore, AI-VAD, FUVAS, WinCLIP |
| First scope | 3 scenes, then all 13 |
| Demo | Gradio |
| Python | Dedicated 3.10 venv |
| Output | Metrics table, heatmaps, timelines, annotated video, report |

---

## 2. Environment and hardware discovery

### 2.1 Machine facts

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060, 12,288 MiB |
| Driver | 621.01 |
| CUDA UMD | 13.5 |
| OS | Windows, PowerShell 5.1 |
| Default Python | 3.13.9 |
| Available Python | 3.10.11 |
| System RAM | **7.9 GB total**, about 5.8 GB free |
| Disk C: | 41.8 GB free at audit time |
| Disk F: | 69 GB free at audit time; project lives here |

### 2.2 Initial package state (system Python 3.13)

| Package | Version |
|---|---|
| torch | 2.10.0+cpu |
| torchvision | 0.25.0 |
| numpy | 2.3.5 |
| scipy | 1.16.3 |
| scikit-learn | 1.8.0 |
| opencv-python | 4.13.0.92 |
| pandas | 3.0.4 |
| pillow | 12.0.0 |
| matplotlib | 3.10.8 |
| tqdm | 4.67.1 |
| transformers | 4.57.6 |
| onnxruntime | 1.31.0 |
| rapidocr-onnxruntime | 1.2.3 |

Missing: Anomalib, Gradio, timm, Lightning, av, TorchCodec, OpenCLIP.

Tools: `tar` present, `7z` absent, `conda` absent, `uv` absent, `winget` present.
Chrome and Edge installed.

### 2.3 The critical discovery: 7.9 GB RAM

The GPU is comfortable (12 GB VRAM), but system RAM is small. This later became the single
biggest constraint for evaluation. GPU memory was never the limiting factor; CPU RAM was.

---

## 3. Tooling decisions and why

### 3.1 Anomalib instead of from-scratch implementations

The user chose the library route. Consequences:

- Fast access to PaDiM, PatchCore, AI-VAD, FUVAS, WinCLIP.
- Built-in metrics, post-processing, thresholding, checkpointing.
- Less "from scratch" signal in interviews, compensated by rigorous experiments,
  ablations, and an honest analysis of library behavior.
- Library-specific issues must be documented (visualizer bug, video batch types,
  TorchCodec dependency).

### 3.2 Python 3.10 venv

Anomalib 2.7.0 metadata says `requires_python >=3.10`, but its published classifiers only
cover 3.10, 3.11, and 3.12. The system Python is 3.13.9, outside the tested range.
Python 3.10.11 was already installed, so a `.venv` was created with it.

### 3.3 CUDA 12.8 wheels

The system PyTorch was CPU-only. The CUDA build was installed from
`https://download.pytorch.org/whl/cu128` first, before Anomalib, to reduce the chance of
Anomalib or Lightning pulling a CPU wheel. The resulting versions were newer than planned:
**torch 2.11.0+cu128** and **torchvision 0.26.0+cu128**.

### 3.4 Install the video extras explicitly

Anomalib did not bring `av`, `open-clip-torch`, or `torchcodec` automatically. They were
installed explicitly:

```powershell
.\.venv\Scripts\python.exe -m pip install av "open-clip-torch" torchcodec
```

### 3.5 Install order

1. Python 3.10 venv.
2. Upgrade pip.
3. CUDA torch/torchvision.
4. Verify CUDA.
5. Anomalib + Gradio.
6. Verify that Anomalib did not replace torch.
7. Video extras.
8. FFmpeg shared DLLs.

---

## 4. Dataset acquisition

### 4.1 CUHK Avenue

- Official page: `http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html`
- Videos: `Avenue_Dataset.zip` (813,227,845 bytes ≈ 775.6 MB).
- Ground truth: `ground_truth_demo.zip` (931,269 bytes ≈ 0.9 MB).
- Contents: 16 training videos (normal), 21 test videos (normal + abnormal),
  47 abnormal events.
- Download method: `curl -L --fail --retry 3`.
- Extraction: `tar -xf` (Windows bsdtar handles zip).

The archive extracted a top-level `Avenue Dataset/` folder. Anomalib expects
`training_videos/` and `testing_videos/` immediately under the root, so those folders were
moved up one level. `ground_truth_demo/` stayed at the root because that is Anomalib's
default `gt_dir`.

### 4.2 UCSD Ped1 / Ped2

- Official page: `http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm`
- Archive: `UCSD_Anomaly_Dataset.tar.gz` (740,306,953 bytes ≈ 706 MB).
- Contents: Ped1 (34 train / 36 test) and Ped2 (16 train / 12 test).
- Videos are stored as folders of frames, not single video files.
- Some test folders have `*_gt/` subfolders with ground-truth masks.
- Extracted to `data/ucsdped/UCSD_Anomaly_Dataset.v1p2/`.

### 4.3 ShanghaiTech Campus — the difficult one

Official page: `https://svip-lab.github.io/dataset/campus_dataset.html`

The official distribution is a folder share on OneDrive:

```
https://1drv.ms/u/s!AjjUqiJZsj8whLt-1ABerTT-9eH9Ag?e=eJbY6Y
```

Inside: `OpenSources > ano_pred_cvpr_2018 > ShanghaiTechCampus > splits`, containing
7 parts of one archive plus a readme.

#### What went wrong

1. **Direct API access failed.**
   - `api.onedrive.com/.../shares/u!...` returned HTTP 400.
   - The short link returned 301 to `onedrive.live.com`, then **403 Forbidden**.
2. **Chrome browser download started but vanished.**
   - Chrome wrote `data\Unconfirmed 267844.crdownload`.
   - It reached at least ~904 MB (measured from actual file bytes; `Get-ChildItem` showed
     stale sizes such as 11.1 MB because Chrome held the file open).
   - At one point the temp file disappeared entirely and no `splits.zip` existed anywhere.
3. **Free Download Manager could not handle it.**
   - Pasting the folder share link produced:
     "Server has returned a web page. Click to open it in the web browser."
   - This is expected: the link is a web page, not a direct file.
   - Right-clicking the OneDrive Download button offered no "copy link".
4. **Chrome's download page trick was plausible but not used in the end** because the
   download became unreliable.

#### What worked

A Hugging Face mirror of the same archive:

```
Dataset: 123zdm/ShanghaiTech
Files:   shanghaitech.tar.gz.aa ... shanghaitech.tar.gz.ag
```

It was downloaded with `huggingface_hub` (already installed as an Anomalib dependency):

```python
from huggingface_hub import hf_hub_download
for s in ["aa","ab","ac","ad","ae","af","ag"]:
    hf_hub_download(repo_id="123zdm/ShanghaiTech",
                    filename=f"shanghaitech.tar.gz.{s}",
                    repo_type="dataset",
                    local_dir="data/shanghaitech_parts")
```

#### Integrity verification

| Part | Bytes |
|---|---|
| `.aa`–`.af` | 1,073,741,824 each |
| `.ag` | 550,511,474 |
| **Total** | **6,992,962,418 (6.51 GB)** |

The parts were joined with `cmd /c copy /b`. The joined file length matched the expected
total byte-for-byte. The archive was extracted with `tar -xzf`.

#### Layout after extraction

The archive extracted into a nested `shanghaitech/` folder. Its contents were moved up so
that `training/` and `testing/` are immediate children of `data/shanghaitech/`.

Actual contents:

| Path | Count | Notes |
|---|---|---|
| `training/videos/` | 330 `.avi` | normal clips |
| `testing/frames/` | 107 folders | `.jpg` frames per test clip |
| `testing/test_frame_mask/` | 107 files | frame-level labels |
| `testing/test_pixel_mask/` | 107 `.npy` files | pixel-level masks |

There was **no** `training/frames/` in the archive, even though some documentation implies
it. Anomalib's ShanghaiTech dataset class uses `training/converted_videos/*.avi` for
training and `testing/frames/<clip>/*.jpg` for testing.

#### Encoding problem and re-encode

The raw ShanghaiTech videos have encoding quirks. The first attempt was to hardlink the raw
videos into `training/converted_videos/` and skip Anomalib's re-encode, because TorchCodec
could decode most of them. This failed during training with:

```
RuntimeError: Requested next frame while there are no more frames left to decode.
```

The hardlinks were removed, and Anomalib's intended `prepare_data()` path was run. It
re-encoded all 330 training videos to XVID. Output: 330 files, about **2.28 GB**, produced
over roughly 15–25 minutes. After that, frame indexing and training were stable.

#### Disk footprint

ShanghaiTech on disk is large because it contains the 7 parts, the joined archive, the
extracted raw data, and the re-encoded videos:

```
data/shanghaitech (total)        ~24.4 GB
```

This is acceptable on F: (about 86 GB free at the time of writing), but the 7 parts and the
joined archive are duplicates and can be deleted after a verified backup/extraction.

---

## 5. Environment build details

### 5.1 Commands that built the environment

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install "anomalib==2.7.0" gradio
.\.venv\Scripts\python.exe -m pip install av "open-clip-torch" torchcodec
```

### 5.2 Verified versions

| Package | Version |
|---|---|
| Python | 3.10.11 |
| pip | 26.2.2 |
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| anomalib | 2.7.0 |
| gradio | 6.31.0 |
| pytorch-lightning | 2.6.6 |
| lightning | 2.6.6 |
| timm | 1.0.30 |
| torchmetrics | 1.9.0 |
| av | 17.1.0 |
| torchcodec | 0.16.0 |
| open-clip-torch | 3.3.0 |
| opencv-python-headless | 5.0.0.93 |
| scikit-learn | 1.7.2 |
| scipy | 1.15.3 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| matplotlib | 3.10.9 |
| tqdm | 4.70.1 |

### 5.3 GPU verification

```python
import torch
print(torch.__version__)                 # 2.11.0+cu128
print(torch.cuda.is_available())         # True
print(torch.cuda.get_device_name(0))     # NVIDIA GeForce RTX 3060
```

### 5.4 Model and data API verification

All of the following imported successfully:

```python
from anomalib.data import ShanghaiTech, Avenue, UCSDped
from anomalib.models import Padim, Patchcore, AiVad, Fuvas, WinClip
from anomalib.engine import Engine
```

Verified constructor parameters (excerpt):

- `ShanghaiTech(root, scene, clip_length_in_frames, frames_between_clips, target_frame, ...)`
- `Avenue(root, gt_dir, clip_length_in_frames, frames_between_clips, ...)`
- `UCSDped(root, category, clip_length_in_frames, frames_between_clips, ...)`
- `Patchcore(backbone, layers, pre_trained, coreset_sampling_ratio, num_neighbors, precision, ...)`
- `Padim(backbone, layers, pre_trained, n_features, ...)`
- `AiVad(box_score_thresh, persons_only, min_bbox_area, max_bbox_overlap,
  enable_foreground_detections, n_velocity_bins, use_velocity_features, use_pose_features,
  use_deep_features, ...)`
- `Fuvas(backbone, layer, pre_trained, spatial_pool, pooling_kernel_size, pca_level, ...)`
- `WinClip(class_name, k_shot, scales, few_shot_source, ...)`

---

## 6. The video-decoding rabbit hole

### 6.1 Symptom

The first Avenue datamodule setup failed with:

```
ImportError: Video decoding capabilities were removed from torchvision and migrated to TorchCodec.
```

torchvision 0.26 no longer decodes video itself.

### 6.2 Installing TorchCodec

```powershell
.\.venv\Scripts\python.exe -m pip install torchcodec
```

Installed: `torchcodec 0.16.0+cpu`.

### 6.3 TorchCodec still failed

```
RuntimeError: Could not load libtorchcodec. Likely causes:
1. FFmpeg is not properly installed in your environment...
On Windows, ensure you've installed the "full-shared" version which ships DLLs.
```

No FFmpeg was installed on the machine. TorchCodec needs **shared** FFmpeg libraries
(`avcodec-*.dll`, `avformat-*.dll`, etc.), not just a static `ffmpeg.exe`.

### 6.4 Portable FFmpeg shared build

Downloaded:

```
https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl-shared.zip
```

Size: 90,941,653 bytes. Extracted to:

```
tools/ffmpeg/ffmpeg-master-latest-win64-gpl-shared/bin
```

It contained FFmpeg 9 libraries, including `avcodec-63.dll`.

### 6.5 The DLL-location trap

First attempt: prepend the FFmpeg `bin` to `PATH` for the process. This worked for a manual
decode test.

Second attempt: copy the DLLs into `.venv\Scripts\` so the app directory would find them.
This **did not** work; TorchCodec still failed to load.

Third attempt (successful): copy the DLLs into the folder that contains the TorchCodec
extension DLLs:

```powershell
Copy-Item "$ffmpegBin\*.dll" ".venv\Lib\site-packages\torchcodec\" -Force
```

Windows resolves dependent DLLs from the directory of the loaded DLL, so this made
`libtorchcodec_core9.dll` find `avcodec-63.dll` and friends.

Verified:

```python
from torchcodec.decoders import VideoDecoder
d = VideoDecoder("data/avenue/training_videos/01.avi")
print(d.metadata.num_frames, d[0].shape)   # 1364, (3, 360, 640)
```

---

## 7. Data preparation details

### 7.1 Avenue

Final layout:

```
data/avenue/
├── training_videos/        # 16 normal clips
├── testing_videos/         # 21 clips
├── training_vol/
├── testing_vol/
├── ground_truth_demo/      # masks, converted from .mat to .png by Anomalib
├── Avenue_Dataset.zip
└── ground_truth_demo.zip
```

Datamodule test result at `clip_length_in_frames=1`:

- train samples: **15,328**
- test samples: **15,324**
- batch image shape: `(B, 3, 480, 856)`

### 7.2 ShanghaiTech

Scene 1 at `clip_length_in_frames=1`, `frames_between_clips=1`:

- train clips: **69,976**
- test clips: **11,894**
- batch image shape: `(8, 3, 480, 856)`

Scene 1 at `frames_between_clips=100`:

- train clips: **47 batches** at batch size 16, i.e. about **700–750 frames**.

### 7.3 Why frame sampling is mandatory for PatchCore

PatchCore stores a feature vector for every patch of every training image.

Approximate calculation for ShanghaiTech scene 1:

- 69,976 frames
- 256×256 input
- WideResNet-50 layer2 stride 8 → 32×32 = 1,024 patches per frame
- concatenated channels: 512 (layer2) + 1,024 (layer3) = 1,536
- storage: 69,976 × 1,024 × 1,536 × 4 bytes ≈ **440 GB**

That is impossible on this machine. The memory bank must be built from a sampled subset of
normal frames (or from lower-resolution/feature-reduced representations). The first real
experiment used every 100th normal frame (~700 frames).

---

## 8. First smoke tests

### 8.1 PatchCore smoke fit (5 batches)

Configuration:

- scene 1
- `clip_length_in_frames=1`
- batch size 8
- `limit_train_batches=5`, `limit_val_batches=2`, `limit_test_batches=2`
- PatchCore, WideResNet-50, layers 2+3, coreset 0.01, neighbors 9

Result:

- GPU used: **True** (Lightning logged `used: True`, device RTX 3060).
- Fit completed, coreset selected over 408 candidate points.
- Test crashed in the visualizer:

```
AttributeError: 'VideoItem' object has no attribute 'image_path'
```

Root cause: Anomalib's `ImageVisualizer` assumes image datasets with `image_path`.
Video datasets produce `VideoItem` with `video_path`. Metrics are not affected.

Fix: pass `visualizer=False` when using image models on video data.

### 8.2 Smoke test after the fix

- Fit + test completed.
- Metrics were 0.0 because the limited test subset contained only normal frames
  ("No positive samples in targets"). Expected for a 2-batch smoke test, not a bug.

---

## 9. First real experiment

### 9.1 Protocol

| Item | Value |
|---|---|
| Model | PatchCore |
| Backbone | WideResNet-50 (`wide_resnet50_2`) |
| Layers | `layer2`, `layer3` |
| Input | 256×256, ImageNet normalization |
| Coreset ratio | 0.01 |
| Neighbors | 9 |
| Precision | float32 |
| Train sampling | every 100th normal frame (~700 frames) |
| Test sampling | every 10th test frame (1,204 frames) |
| Batch size | 16 |
| Workers | 0 |
| Max epochs | 1 (feature fit, no gradient training) |
| Threshold | adaptive, from validation split |
| Visualizer | disabled |

### 9.2 First attempt with full test

- Fit: **239.5 s**
- Test inference: **744 batches / 11,894 frames**, **33 min 52 s**
- Then AUROC computation crashed:

```
DefaultCPUAllocator: not enough memory: you tried to allocate 6,235,881,472 bytes
```

Cause: TorchMetrics accumulated pixel predictions for all frames and tried to sort them.
With 11,894 frames and 256×256 maps, that is about 780 million values.

### 9.3 Re-test from checkpoint with stride 10

The fit checkpoint had been saved:

```
results/patchcore/scene01/Patchcore/ShanghaiTech/v0/weights/lightning/model.ckpt
```

It was reloaded with `Patchcore.load_from_checkpoint(...)`, and the test datamodule was
created with `frames_between_clips=10` (1,204 test frames).

Result:

| Metric | Value |
|---|---|
| image_AUROC | **0.7853** |
| image_F1Score | **0.6716** |
| pixel_AUROC | **0.9887** |
| pixel_F1Score | **0.5948** |

Test time: **379.9 s** (about 3.2 frames/s end-to-end including metric computation).

### 9.4 Interpretation

- Pixel localization is strong (0.989 pixel AUROC).
- Image-level detection is moderate (0.785 image AUROC).
- This is plausible for a frame-wise image model on ShanghaiTech: motion anomalies such as
  running/chasing are hard without temporal reasoning. This motivates AI-VAD and FUVAS.
- Numbers are not directly comparable to published video results because the test set was
  subsampled and the memory bank is small.

### 9.5 First annotated demo video

A renderer was written (`src/render_video.py`) to turn the fitted PatchCore checkpoint into a
watchable MP4. It runs frame-by-frame inference, thresholds the anomaly map, finds contours,
and draws red boxes with a score banner.

Source clip: **`01_0136`** (scene 1, 529 frames, 856×480, 311 ground-truth anomalous frames).

The first render used the checkpoint thresholds (pixel 48.84, image 46.27). It produced
many small boxes on normal frames (for example a backpack at frame 62/80), so a threshold
sweep was done against the clip's frame mask:

| Image threshold | Precision | Recall | F1 |
|---|---|---|---|
| 46.3 (checkpoint) | 0.66 | 1.00 | 0.80 |
| 52 | 0.89 | 0.93 | 0.91 |
| 54 | 0.97 | 0.89 | 0.93 |
| 58 | 0.99 | 0.79 | 0.88 |

Final display settings: **pixel threshold 55**, **image threshold 58**, **min box area 200**,
boxes drawn only when the frame is labelled ANOMALY. This removed the small false boxes while
keeping the truck. The two remaining high-score frames with GT=0 (271, 323) visibly contain
the truck, so they are likely sparse annotation rather than model errors.

Result:

- 529 frames rendered in 19.8 s (~26.7 fps)
- 247 ANOMALY frames, 245 with boxes
- first/last boxed frame: 169 / 431
- max score 82.35 (truck)
- 856×480 @ 10 fps, 4.4 MB

Artifacts:

```
results/videos/shanghaitech_scene01_01_0136_annotated.mp4
results/videos/shanghaitech_scene01_01_0136_annotated.scores.csv
results/videos/01_0136_samples/*.jpg
```

This is a visualization-only result; it does not change the metric protocol. The display
thresholds are tuned for presentation, while reported metrics use the checkpoint thresholds.

### 9.6 People-only AI-VAD demo

The first demo video was a truck (PatchCore). The user then asked for a **people-only**
video. Testing the fight clip `01_0026` with PatchCore gave scores of 44-56 and **no boxes**;
a scan of all scene-1 clips showed the frame-wise model only fires on vehicles. This is the
motion-blindness the project anticipated.

The video-native **AI-VAD** was therefore fitted with `persons_only=True`:

- Model: Mask R-CNN person detection + Keypoint R-CNN + RAFT flow + CLIP ViT-B/16 +
  velocity/pose/deep density estimators.
- Data: scene 1, clips of 2 frames, every 100th frame (738 training clips).
- Fit time: 5 min 02 s; 2,262 person detections.
- Checkpoint: `results/aivad/scene01/.../model.ckpt` (762.6 MB).

On the fight clip, raw scores were ~1.04 on a normal frame and 1.41-1.63 during the fight.
Frame-level thresholds were noisy, so the renderer adds **temporal smoothing (window 15)**.
A causal moving average plus threshold 1.3 and pixel threshold 1.1 produced a clean event
segment; false positives clustered **outside** the fight (frames 23-35, 141-202, 398-403),
so the demo was cropped to frames **260-420**.

Two practical issues surfaced:

1. **Empty-region bug:** AI-VAD crashes with
   `IndexError: amax(): Expected reduction dim 0 to have non-zero size` when no person is
   detected. The renderer catches it and treats the frame as normal.
2. **Causal smoothing tail:** the banner can lag the boxes; requiring at least one box fixed
   the mismatch.

Output:

- `results/videos/shanghaitech_scene01_01_0026_people_annotated.mp4`
- 161 frames (260-420), 10 fps, 856×480, 1.1 MB
- 91 frames with red person boxes; first/last boxed frame 311/405; max smoothed score 2.40
- Frames are labelled `PEOPLE ANOMALY` only when boxes are present

The renderer was corrected after inspection showed a walker boxed before the fight. The
initial implementation ranked each detector box by the maximum anomaly-map value inside
it. The final implementation uses AI-VAD's density-estimator per-region `box_scores`;
frame 263 is no longer boxed, while the fight boxes remain.

This is a demonstration of the video-native branch of the project, not a benchmark result.

### 9.7 Product-hardening decision

The walking boxes before the fight showed why a product cannot treat each AI-VAD frame as an
independent alarm. The next design keeps AI-VAD but adds a deployment-matched temporal gap,
velocity/pose/deep diagnostics, per-person region scores, lightweight IoU tracking, rolling
track histories, persistence, hysteresis, debounce, cooldown, and camera-specific calibration.
The implementation is in `src/render_people_video.py`; held-out normal calibration is in
`src/calibrate_people_thresholds.py`.

This is deliberately a prototype architecture. AI-VAD can flag unusual behavior, but a
reliable “fight detector” may still require a supervised verifier trained with hard negatives
such as walking, running, waving, crowds, and camera-specific motion.

The first independent calibration pass used normal training footage from scenes 2 and 3.
At gap 5 its 99.5th-percentile image/region thresholds were 2.937/2.253. A separate
screen over visually reviewed mask-zero intervals from scene-1 test clips reached maxima
of 4.288/3.050, confirming that walking and crowd motion are useful hard negatives rather
than evidence that the fight event is correctly identified. These intervals are provisional
until human review is complete; no hard-negative refit has been claimed.

---

## 10. Hyperparameters used (first run)

### PatchCore

| Parameter | Value |
|---|---|
| `backbone` | `wide_resnet50_2` |
| `layers` | `["layer2", "layer3"]` |
| `pre_trained` | True (default) |
| `coreset_sampling_ratio` | 0.01 |
| `num_neighbors` | 9 |
| `precision` | float32 |
| `image_size` | 256×256 |
| `train_batch_size` | 16 |
| `eval_batch_size` | 16 |
| `num_workers` | 0 |
| `clip_length_in_frames` | 1 |
| `frames_between_clips` (train) | 100 |
| `frames_between_clips` (test) | 10 |
| `max_epochs` | 1 |
| `visualizer` | False |

### Datamodule / evaluation

| Parameter | Value |
|---|---|
| `target_frame` | last (default) |
| `val_split_mode` | same as test (default) |
| `test_split_mode` | from directory (default) |
| threshold | adaptive from validation |
| seed | not fixed yet |

---

## 11. Challenges faced (complete list)

Every issue is documented in detail in `docs/CHALLENGES.md`. Summary:

| # | Challenge | Outcome |
|---|---|---|
| 1 | User wanted video; MVTec-AD is images only | Pivoted to video benchmarks |
| 2 | Anomalib 2.7 outside Python 3.13 tested range | Created Python 3.10 venv |
| 3 | System PyTorch was CPU-only | Installed CUDA 12.8 wheels |
| 4 | OneDrive blocked automated ShanghaiTech download (403) | Used Hugging Face mirror |
| 5 | Chrome download disappeared; temp file sizes looked stale | Abandoned browser route |
| 6 | FDM could not download a OneDrive folder share | Explained folder vs direct link |
| 7 | torchvision 0.26 removed video decoding | Installed TorchCodec |
| 8 | TorchCodec could not find FFmpeg shared DLLs | Portable FFmpeg shared build |
| 9 | DLLs in `.venv/Scripts` were not found | Copied DLLs into `site-packages/torchcodec` |
| 10 | Avenue had an extra top-level folder | Moved dataset folders up |
| 11 | ShanghaiTech had no `training/frames` | Used `converted_videos` as Anomalib expects |
| 12 | Raw ShanghaiTech videos failed frame indexing | Let Anomalib re-encode to XVID |
| 13 | Image visualizer crashed on video batches | `visualizer=False` |
| 14 | PatchCore would need ~440 GB for 70k frames | Subsampled normal frames |
| 15 | Full-test pixel AUROC ran out of RAM (7.9 GB) | Subsample test frames / plan streaming metrics |
| 16 | Smoke test metrics were zero | Expected: limited test subset had no anomalies |
| 17 | DataLoader slow with `num_workers=0` | Known; can try 2–3 workers |
| 18 | Threshold warning: validation set has no anomalies | Threshold derived from normal scores; document it |
| 19 | HF cache symlink warning on Windows | Non-fatal; can enable Developer Mode |
| 20 | `triton not found` warning | Non-fatal on Windows; FLOP counting disabled |
| 21 | Many timm/deprecation warnings | Non-fatal; version drift in Anomalib |
| 22 | Dataset licenses restrict commercial use | Documented; research only |

---

## 12. What still needs tuning

No systematic tuning has been done yet. The first run is a baseline with reasonable
defaults. The following are planned and necessary.

### 12.1 Memory-bank size (most important)

- Current: every 100th normal frame (~700 frames).
- Alternatives: 400, 1,000, 2,000 frames.
- Trade-off: larger bank → better coverage → more RAM/time.
- Constraint: 7.9 GB system RAM.

### 12.2 Coreset ratio

- Current: 0.01.
- Alternatives: 0.001, 0.005, 0.01, 0.05, 0.1.
- Trade-off: smaller memory bank at test time vs AUROC and localization.

### 12.3 Backbone

- Current: `wide_resnet50_2`.
- Alternatives: `resnet50`, larger timm models, ViT/DINOv2.
- Trade-off: accuracy vs speed and memory.

### 12.4 Feature layers

- Current: `layer2 + layer3`.
- Alternatives: `layer2`, `layer3`, `layer1 + layer2 + layer3`.
- Trade-off: spatial resolution vs semantic depth.

### 12.5 Input resolution

- Current: 256.
- Alternatives: 224, 320.
- Resolution directly controls pixel-metric memory and inference time.

### 12.6 Neighbors

- Current: 9.
- Alternatives: 1, 5, 9.
- Affects score smoothing and speed.

### 12.7 Test sampling for evaluation

- Current: every 10th frame (1,204).
- Final benchmark should use every frame or a principled subset.
- Best solution: streaming/per-video AUROC that does not hold all pixel predictions.

### 12.8 PaDiM-specific

- `n_features`: 100 vs 550.
- covariance regularization (epsilon): 0.01 / 0.1 / 0.5.
- layers/backbone.

### 12.9 AI-VAD-specific

From the verified signature: `box_score_thresh`, `persons_only`, `min_bbox_area`,
`max_bbox_overlap`, `enable_foreground_detections`, `foreground_kernel_size`,
`foreground_binary_threshold`, `n_velocity_bins`, `use_velocity_features`,
`use_pose_features`, `use_deep_features`, `n_components_velocity`, `n_neighbors_pose`,
`n_neighbors_deep`.

Important because AI-VAD is the first true video-native model in the lineup.

### 12.10 FUVAS-specific

- `backbone`, `layer`, `spatial_pool`, `pooling_kernel_size`, `pca_level`.
- Few-shot `k`: 1, 2, 4, 8.

### 12.11 WinCLIP-specific

- `scales`: 1, 2, 3.
- Prompt templates (normal vs abnormal).
- `class_name` handling per scene.

### 12.12 Temporal layer

- Smoothing window: 1, 5, 11, 21 frames.
- Moving average vs median vs max.
- Hysteresis thresholds and debounce length.

### 12.13 Seeds and reproducibility

- Set seeds for Python, NumPy, PyTorch.
- Save the exact config with every run.
- Record package versions.

---

## 13. Current state

### Ready

- Environment (GPU verified).
- All datasets.
- Data pipelines for Avenue and ShanghaiTech.
- One real PatchCore result.
- Documentation set.

### Not ready

- `src/render_video.py` and `src/render_people_video.py` are implemented; the general
  runner, `configs/`, and `app/` remain pending.
- No PaDiM/FUVAS/WinCLIP benchmark runs yet; AI-VAD has a scene-1 fit and people-only demo.
- Temporal product layer is implemented in the AI-VAD renderer; benchmark event metrics remain.
- No Gradio demo.
- No memory-safe full-test evaluation.
- No hyperparameter ablations.
- No report.

---

## 14. Next steps (ordered)

1. Write `src/run.py` — config-driven runner.
2. Implement per-scene memory-bank sampling (target ~700 frames, adjustable).
3. Implement memory-safe evaluation:
   - image AUROC on all test frames (cheap),
   - pixel AUROC on a fixed subsample or streaming per video.
4. Save and reuse checkpoints.
5. Run PaDiM scene 1 with the same protocol.
6. Run PaDiM + PatchCore on scenes 1–3.
7. Run AI-VAD on scenes 1–3 (scene 1 fit/demo is complete; scenes 2–3 remain).
8. Add temporal smoothing and event metrics.
9. Run FUVAS few-shot, then WinCLIP zero-shot.
10. Extend to all 13 scenes + Avenue.
11. Gradio demo + ONNX export + FPS table.
12. Final report and results table.

---

## 15. Lessons learned

1. **Check the dataset first.** "Latest industrial anomaly detection" and "video" are not
   compatible with the standard MVTec benchmark. Verify modality before planning.
2. **Check the Python support matrix.** Anomalib's metadata was permissive, but classifiers
   stopped at 3.12.
3. **Install CUDA torch first, then the framework, then re-verify.** Dependency graphs can
   silently replace GPU wheels.
4. **Video decoding is a dependency minefield.** torchvision 0.26 → TorchCodec → FFmpeg
   shared DLLs → correct DLL directory.
5. **Windows DLL search behavior matters.** App-directory copies did not work; the DLL's
   own directory did.
6. **Official large-dataset links can be unreliable.** A verified mirror plus byte-size
   checks saved the day.
7. **Frame counts are not free.** A 70k-frame scene is not a 70k-image dataset for
   memory-bank methods; it is two orders of magnitude larger.
8. **System RAM, not VRAM, was the bottleneck.** The GPU was never the problem; metric
   accumulation was.
9. **Document the exact protocol.** Subsampled training and evaluation must be stated or
   the numbers are misleading.
10. **Smoke-test metrics can legitimately be zero.** A limited test subset may contain no
    positives; that is not a pipeline failure.

---

## 16. Timeline (2026-11-02)

| Time (approx) | Event |
|---|---|
| 10:34 | GPU audit (`nvidia-smi`) |
| 11:00 | Avenue + UCSD downloaded and extracted |
| 12:26–12:44 | ShanghaiTech browser download issues; temp file observed |
| 12:44 | Python 3.10 venv created; pip upgraded |
| 12:44–13:02 | CUDA torch, Anomalib, Gradio installed |
| 13:02 | GPU and model imports verified |
| 13:42 | Avenue datamodule failed: TorchCodec missing |
| 13:44 | FFmpeg shared build installed; DLL fix |
| 13:46 | ShanghaiTech datamodule verified (69,976 / 11,894) |
| 13:50–14:14 | ShanghaiTech re-encode to XVID (330 clips) |
| 14:15 | PatchCore smoke test (visualizer fix) |
| 14:58 | Real scene-1 run; full test OOM at AUROC |
| 15:05 | Re-test from checkpoint with stride 10; metrics obtained |
| 15:15+ | Documentation updated; journey authored |

---

## 17. Appendix: key commands

### Build environment

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install "anomalib==2.7.0" gradio
.\.venv\Scripts\python.exe -m pip install av "open-clip-torch" torchcodec
```

### Verify GPU

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Join ShanghaiTech parts

```powershell
cmd /c copy /b shanghaitech.tar.gz.aa+shanghaitech.tar.gz.ab+shanghaitech.tar.gz.ac+shanghaitech.tar.gz.ad+shanghaitech.tar.gz.ae+shanghaitech.tar.gz.af+shanghaitech.tar.gz.ag shanghaitech.tar.gz
```

### Extract

```powershell
tar -xzf shanghaitech.tar.gz -C data\shanghaitech
```

### Fix FFmpeg DLLs for TorchCodec

```powershell
Copy-Item "<ffmpeg-bin>\*.dll" ".venv\Lib\site-packages\torchcodec\" -Force
```

### Run a PatchCore experiment (conceptual)

```python
from anomalib.data import ShanghaiTech
from anomalib.models import Patchcore
from anomalib.engine import Engine

dm_train = ShanghaiTech(root="data/shanghaitech", scene=1, clip_length_in_frames=1,
                        frames_between_clips=100, train_batch_size=16, num_workers=0)
dm_test = ShanghaiTech(root="data/shanghaitech", scene=1, clip_length_in_frames=1,
                       frames_between_clips=10, eval_batch_size=16, num_workers=0)

model = Patchcore(backbone="wide_resnet50_2", layers=["layer2","layer3"],
                  coreset_sampling_ratio=0.01, num_neighbors=9, visualizer=False)
engine = Engine(max_epochs=1, accelerator="gpu", devices=1,
                default_root_dir="results/patchcore/scene01")
engine.fit(model=model, datamodule=dm_train)
engine.test(model=model, datamodule=dm_test)
```

---

## 18. Appendix: file and artifact inventory

### Documentation

| File | Purpose |
|---|---|
| `README.md` | Entry point, status, first results |
| `PROJECT_GUIDE.md` | Reference manual, plan, runbooks |
| `docs/PROGRESS.md` | Running log |
| `docs/JOURNEY.md` | This complete journey |
| `docs/DECISIONS.md` | Decision records |
| `docs/CHALLENGES.md` | Problem catalog |
| `docs/EXPERIMENTS.md` | Experiment and tuning log |
| `docs/Project_Journey.pdf` | PDF build of the journey docs |

### Code and config

| Path | State |
|---|---|
| `configs/` | Empty (planned) |
| `src/` | PatchCore and AI-VAD renderers; benchmark runner planned |
| `app/` | Empty (planned) |
| `tests/` | Empty (planned) |
| `requirements.txt` | Written with verified versions |
| `results/summary.csv` | First result row |

### Data

| Path | Size / contents |
|---|---|
| `data/avenue/` | ~1.53 GB |
| `data/ucsdped/` | ~1.77 GB |
| `data/shanghaitech/` | ~24.4 GB (parts, joined archive, raw, converted) |
| `data/shanghaitech_parts/` | 7 archive parts + joined archive |

### Models / results

| Path | Contents |
|---|---|
| `results/patchcore/scene01/.../model.ckpt` | Fitted PatchCore checkpoint (~139.5 MB) |
| `results/smoke_patchcore/.../model.ckpt` | Smoke-test checkpoints |

---

## 19. Appendix: glossary

| Term | Meaning |
|---|---|
| AUROC | Area under the ROC curve; threshold-free ranking metric |
| Pixel AUROC | AUROC computed per pixel for localization |
| F1 | Harmonic mean of precision and recall; needs a threshold |
| Coreset | A representative subsample of a larger set |
| Memory bank | Stored normal feature vectors used for nearest-neighbor scoring |
| Frame-wise | Treat each frame independently |
| Video-native | Use temporal information such as motion or clips |
| Few-shot | Learn from a small number of normal examples |
| Zero-shot | No target-dataset training |
| Temporal smoothing | Aggregate frame scores over time |
| TorchCodec | PyTorch video decoding library used by torchvision 0.26+ |
| FFmpeg shared | FFmpeg build that ships DLLs, required by TorchCodec |
| Anomalib | Intel's anomaly-detection library |
| ShanghaiTech | 13-scene campus video anomaly benchmark |
| Avenue | CUHK Avenue abnormal-event benchmark |
| UCSD Ped | UCSD pedestrian anomaly benchmark |
