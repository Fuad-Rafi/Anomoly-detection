# Challenge Catalog

Every notable problem encountered during the project, with the exact symptom, the root
cause, the fix applied, the current status, and the lesson. Ordered roughly by when it
appeared.

---

## C-01 — "Industrial anomaly detection" has no standard video benchmark

- **Symptom:** The user asked for video, but the planned datasets were MVTec-AD / MVTec AD 2.
- **Root cause:** MVTec-AD and MVTec AD 2 are still-image datasets. They contain no video
  clips.
- **Evidence:** Official dataset pages; Anomalib video datamodules are only `Avenue`,
  `ShanghaiTech`, `UCSDped`.
- **Fix:** Pivoted to surveillance video anomaly detection; user accepted the domain change.
- **Status:** Resolved.
- **Lesson:** Verify the data modality before committing to a dataset/method plan.

---

## C-02 — Anomalib 2.7.0 outside the system Python's tested range

- **Symptom:** System Python is 3.13.9. Anomalib 2.7.0 classifiers stop at 3.12.
- **Root cause:** Metadata says `requires_python >=3.10`, but published classifiers only
  cover 3.10, 3.11, 3.12.
- **Fix:** Created `.venv` with the already-installed Python 3.10.11.
- **Status:** Resolved.
- **Lesson:** Read classifiers, not only `requires_python`.

---

## C-03 — System PyTorch was CPU-only

- **Symptom:** `torch.__version__` = `2.10.0+cpu`; `torch.cuda.is_available()` = False.
- **Root cause:** The default Python environment had a CPU wheel.
- **Fix:** Installed torch/torchvision from the CUDA 12.8 index first; verified CUDA True.
- **Status:** Resolved.
- **Lesson:** Install the GPU wheel before frameworks that depend on PyTorch.

---

## C-04 — The official ShanghaiTech OneDrive link blocked automated downloads

- **Symptom:**
  - `api.onedrive.com/.../shares/u!...` returned HTTP 400.
  - Short link returned 301 to `onedrive.live.com`, then 403 Forbidden.
- **Root cause:** The share is a web folder and/or requires a browser session; the API forms
  tried were not accepted.
- **Fix:** Used a Hugging Face mirror of the same 7-part archive.
- **Status:** Resolved via mirror.
- **Lesson:** Have a verified mirror plan for large public datasets.

---

## C-05 — Chrome download vanished, and file sizes looked wrong

- **Symptom:**
  - `splits.zip` download reached at least ~904 MB.
  - `data\Unconfirmed 267844.crdownload` later disappeared; no `splits.zip` existed.
  - `Get-ChildItem` reported 11.1 MB while `[System.IO.FileInfo].Length` reported hundreds
    of MB for the same file.
- **Root cause:** Chrome held the file open and filesystem metadata was stale; the download
  was ultimately canceled/discarded.
- **Fix:** Abandoned the browser route; used the HF mirror. For diagnostics, read
  `[System.IO.FileInfo]::new(path).Length` instead of cached directory listings.
- **Status:** Resolved.
- **Lesson:** Do not trust a single size reading for an in-progress browser download.

---

## C-06 — Download manager could not handle the OneDrive folder

- **Symptom:** FDM: "Server has returned a web page. Click to open it in the web browser."
- **Root cause:** The pasted URL is a folder share page, not a direct file URL. Folder shares
  cannot be enumerated by a simple download manager.
- **Fix:** Explained direct-link vs folder-link; recommended JDownloader 2 for folders, or
  the HF mirror (chosen).
- **Status:** Resolved.
- **Lesson:** Download managers need direct file URLs; folder shares need a browser or a
  specialized mirror.

---

## C-07 — torchvision 0.26 removed video decoding

- **Symptom:**
  ```
  ImportError: Video decoding capabilities were removed from torchvision and migrated to TorchCodec.
  ```
- **Root cause:** torchvision 0.26 delegates video decoding to TorchCodec.
- **Fix:** `pip install torchcodec`.
- **Status:** Resolved (but see C-08).
- **Lesson:** Major version upgrades can remove long-standing functionality.

---

## C-08 — TorchCodec could not find FFmpeg shared libraries

- **Symptom:**
  ```
  RuntimeError: Could not load libtorchcodec. Likely causes:
  1. FFmpeg is not properly installed...
  On Windows, ensure you've installed the "full-shared" version which ships DLLs.
  ```
- **Root cause:** No FFmpeg was installed; TorchCodec needs shared `avcodec-*.dll`,
  `avformat-*.dll`, etc. A static `ffmpeg.exe` is not sufficient.
- **Fix:** Downloaded the BtbN portable shared build (90,941,653 bytes), FFmpeg 9 /
  `avcodec-63.dll`.
- **Status:** Resolved (with C-09).
- **Lesson:** "FFmpeg installed" can mean a static binary; shared libraries are different.

---

## C-09 — DLL directory mattered: `.venv/Scripts` did not work

- **Symptom:** After copying FFmpeg DLLs into `.venv\Scripts`, TorchCodec still failed with
  `Could not find module ... or one of its dependencies`.
- **Root cause:** Windows resolves dependent DLLs relative to the loaded DLL's directory,
  not reliably from the application directory in this setup.
- **Fix:** Copied the DLLs into `.venv\Lib\site-packages\torchcodec\` (next to
  `libtorchcodec_core9.dll`).
- **Status:** Resolved.
- **Lesson:** Put dependent DLLs next to the extension DLL that needs them.

---

## C-10 — Avenue extracted with an extra top-level folder

- **Symptom:** Anomalib's `Avenue` datamodule expects `training_videos/` and
  `testing_videos/` directly under the root, but the zip created `Avenue Dataset/`.
- **Root cause:** Archive layout differs from Anomalib's assumption.
- **Fix:** Moved `training_videos`, `testing_videos`, `training_vol`, `testing_vol` up one
  level; left `ground_truth_demo/` at the root.
- **Status:** Resolved.
- **Lesson:** Normalize dataset layout to the library's expected structure before debugging
  the model.

---

## C-11 — ShanghaiTech had no `training/frames/`

- **Symptom:** Documentation implied frame folders, but the archive only contained
  `training/videos/*.avi`.
- **Root cause:** Dataset distribution difference.
- **Fix:** Confirmed Anomalib training uses `training/converted_videos/*.avi`; testing uses
  `testing/frames/<clip>/*.jpg`.
- **Status:** Resolved.
- **Lesson:** Read the dataset class source, not just the docs.

---

## C-12 — Raw ShanghaiTech videos failed frame indexing

- **Symptom:**
  ```
  RuntimeError: Requested next frame while there are no more frames left to decode.
  ```
- **Root cause:** Raw ShanghaiTech videos have encoding quirks. Hardlinking them into
  `converted_videos/` skipped Anomalib's intended re-encode and exposed the issue.
- **Fix:** Removed the hardlinks; ran `prepare_data()`, which re-encoded all 330 training
  videos to XVID.
- **Status:** Resolved.
- **Lesson:** Respect the library's data preparation for known-problematic encodings.

---

## C-13 — Image visualizer crashed on video batches

- **Symptom:**
  ```
  AttributeError: 'VideoItem' object has no attribute 'image_path'
  ```
- **Root cause:** Anomalib's `ImageVisualizer` assumes image datasets; video batches carry
  `video_path`.
- **Fix:** Pass `visualizer=False` to image models used on video data.
- **Status:** Resolved (workaround).
- **Lesson:** Metrics and visualization are separate concerns; disable the incompatible one.

---

## C-14 — PatchCore memory explosion with 70k frames

- **Symptom:** Estimated ~440 GB of patch features for scene 1.
- **Root cause:** PatchCore stores every patch embedding; 69,976 frames × 1,024 patches ×
  1,536 dims × 4 bytes.
- **Fix:** Build the memory bank from every 100th normal frame (~700 frames).
- **Status:** Mitigated; needs principled ablation.
- **Lesson:** Frame-wise methods scale with frames, not videos; sample deliberately.

---

## C-15 — Full-test pixel AUROC ran out of RAM

- **Symptom:**
  ```
  DefaultCPUAllocator: not enough memory: you tried to allocate 6,235,881,472 bytes
  ```
  after 33 min of successful inference on 11,894 frames.
- **Root cause:** TorchMetrics accumulated ~780 million pixel predictions and sorted them;
  the machine has 7.9 GB RAM.
- **Fix (temporary):** Re-tested from checkpoint with every 10th frame (1,204 frames).
- **Status:** Open engineering issue for the full benchmark.
- **Lesson:** The bottleneck is metric accumulation and CPU RAM, not GPU or inference.

---

## C-16 — Smoke-test metrics were all zero

- **Symptom:** `image_AUROC = 0`, `pixel_AUROC = 0` after a limited test.
- **Root cause:**
  ```
  No positive samples in targets, true positive value should be meaningless.
  ```
  The 2-batch test subset contained only normal frames.
- **Fix:** None needed; this was expected for the smoke test.
- **Status:** Understood.
- **Lesson:** A limited test subset can legitimately yield zero metrics.

---

## C-17 — Validation split contained no anomalies

- **Symptom:**
  "The validation set does not contain any anomalous images. As a result, the adaptive
  threshold will take the value of the highest anomaly score observed in the normal
  validation images..."
- **Root cause:** Default validation split draws from test; with a small/limited split it
  may contain only normal frames.
- **Fix:** Accept for AUROC (threshold-free); for F1, document that the threshold is derived
  from normal scores. Improve the validation protocol later.
- **Status:** Open protocol decision.
- **Lesson:** Threshold selection needs an explicit, anomaly-containing validation split or
  a different rule.

---

## C-18 — Data loading is slow with `num_workers=0`

- **Symptom:** Test inference ~5.9 fps; Lightning warns about missing workers.
- **Root cause:** Video decoding and JPEG loading run in the main process.
- **Fix (proposed):** Try `num_workers=2–3`; measure stability and RAM.
- **Status:** Open optimisation.
- **Lesson:** Windows dataloading needs care; workers can help but cost RAM.

---

## C-19 — Hugging Face cache symlink warning on Windows

- **Symptom:**
  "huggingface_hub cache-system uses symlinks by default ... your machine does not support
  them."
- **Root cause:** Windows symlink support requires Developer Mode or admin.
- **Fix:** None needed; cache still works with extra disk usage.
- **Status:** Non-blocking.
- **Lesson:** Warnings are not always errors.

---

## C-20 — `triton not found` warning

- **Symptom:** "triton not found; flop counting will not work for triton kernels."
- **Root cause:** Triton is not available on this Windows setup.
- **Fix:** None needed; FLOP counting is optional.
- **Status:** Non-blocking.
- **Lesson:** Ignore optional tooling warnings.

---

## C-21 — Deprecation warnings from timm/Anomalib

- **Symptom:** `timm.models.layers` deprecation; `anomalib.models.components.dinov2`
  deprecation.
- **Root cause:** Version drift inside Anomalib.
- **Fix:** None needed now; pin versions.
- **Status:** Non-blocking.
- **Lesson:** Pin versions for reproducibility.

---

## C-22 — Dataset licenses restrict commercial use

- **Symptom:** ShanghaiTech is BSD-2-Clause; Avenue is academic-only; UCSD is research-use.
- **Root cause:** Standard academic dataset licensing.
- **Fix:** Documented in README and guide; project treated as research/portfolio.
- **Status:** Documented.
- **Lesson:** Always state dataset licenses.

---

## C-23 — `Get-ChildItem` showed stale sizes for an open file

- **Symptom:** The same `.crdownload` appeared as 11.1 MB and ~837 MB in different commands.
- **Root cause:** Directory metadata caching while Chrome held the file open.
- **Fix:** Used `[System.IO.FileInfo]::new(path).Length` and a 4-second delta measurement.
- **Status:** Resolved.
- **Lesson:** For live files, query the file object directly and sample twice.

---

## C-24 — Re-encode output directories were missing

- **Symptom:** `training/converted_videos` and `training/frames` did not exist in the archive.
- **Root cause:** The distribution only ships `training/videos`.
- **Fix:** Created `training/frames` and `training/converted_videos` directories; let
  `prepare_data()` populate converted videos.
- **Status:** Resolved.
- **Lesson:** Create expected directories before assuming a library will.

---

## C-25 — Training "instant start" after download was not realistic

- **Symptom:** Expectation: start training immediately when the archive finished.
- **Root cause:** Unavoidable steps: join/extract 6.5 GB, re-encode 330 videos, build the
  venv, install 2.5 GB of torch, configure TorchCodec/FFmpeg, and produce a memory bank.
- **Fix:** Parallelized environment setup with the download, and prepared a watcher-style
  runbook. Communicated the real prep window.
- **Status:** Understood.
- **Lesson:** "Instant" is a sequence of prerequisites; list them explicitly.

---

## Open technical issues (summary)

## C-26 — Walking false positives before a fight

- **Symptom:** Boxes appeared on walking people around seconds 6-8, then returned during
  the fight.
- **Likely causes:** Sparse stride-100 fitting, mismatch between training and inference
  temporal gaps, per-frame decisions, and thresholds tuned on one demonstration clip.
- **Fix direction:** Add `--frame-gap`, log component and region scores, use IoU track
  histories with persistence and hysteresis, and calibrate on held-out normal walking data.
- **Status:** Mitigation implemented in the renderer; calibration and hard-negative mining
  remain product validation work.
- **Lesson:** A CCTV alarm should be calibrated and scored per tracked subject over time.

## C-27 - Mixed normal and anomalous candidate clips

- **Symptom:** Candidate ShanghaiTech clips contain walking/crowd footage but also labelled
  anomaly intervals.
- **Root cause:** Test clips are event sequences, not wholly normal calibration videos.
- **Fix:** Restrict screening to mask-zero intervals and retain the interval definitions in
  the calibration output.
- **Status:** Initial screening complete; human interval review and hard-negative refit remain
  pending.
- **Lesson:** A low-positive clip is not automatically a valid negative.

| Issue | Priority | Next action |
|---|---|---|
| Memory-safe full-test evaluation | High | Stream metrics per video or use two-tier evaluation |
| Memory-bank size ablation | High | 400 / 700 / 1,000 / 2,000 frames |
| Validation threshold protocol | High | Build a validation split with anomalies or define a rule |
| Data loading speed | Medium | Try `num_workers=2–3` |
| Reproducible seeds | Medium | Seed Python/NumPy/PyTorch, log configs |
| Visual heatmaps | Medium | Generate overlays ourselves (visualizer disabled) |
| FUVAS/WinCLIP resource use | Medium | Test at low resolution first |
| Disk cleanup | Low | Remove duplicate split parts after final verification |
