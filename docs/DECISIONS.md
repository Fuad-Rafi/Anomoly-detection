# Decision Log (ADR-style)

Every significant decision made during the project, with context, alternatives, the choice
made, and its consequences. Status values: **Accepted**, **Superseded**, **Provisional**.

---

## D-01 — Choose unsupervised anomaly detection as the project type

- **Date:** 2026-11-02
- **Status:** Accepted
- **Context:** The user wanted a computer-vision project using current technology, not too
  hard, that demonstrates a strong ML/vision foundation.
- **Alternatives:** Detection + tracking + counting, text-prompted segmentation with
  Grounding DINO + SAM 2, monocular depth, CLIP retrieval, VLM fine-tuning.
- **Decision:** Unsupervised anomaly detection, because it teaches one-class learning,
  representation quality, localization, and rigorous evaluation — all common in industry.
- **Consequences:** Requires careful evaluation protocol (AUROC, pixel metrics, threshold
  rules). Strong interview signal. No need to collect defect labels.

---

## D-02 — Use Anomalib instead of from-scratch implementations

- **Date:** 2026-11-02
- **Status:** Accepted
- **Context:** The user chose a library-based route for speed and reproducibility.
- **Alternatives:** Implement PaDiM/PatchCore from scratch; hybrid; use Anomalib only as a
  cross-check.
- **Decision:** Anomalib 2.7.0.
- **Consequences:**
  - Fast access to PaDiM, PatchCore, AI-VAD, FUVAS, WinCLIP.
  - Built-in metrics, thresholding, checkpointing, Engine.
  - Less "built from scratch" signal; compensated by experiments, ablations, and analysis.
  - Must document library behavior and bugs (visualizer, video batches, TorchCodec).

---

## D-03 — Dedicated Python 3.10 virtual environment

- **Date:** 2026-11-02
- **Status:** Accepted
- **Context:** System default is Python 3.13.9. Anomalib 2.7.0 declares `>=3.10` but its
  classifiers cover only 3.10–3.12.
- **Alternatives:** Use 3.13 and risk unsupported dependencies; install 3.12; use conda.
- **Decision:** Use the already-installed Python 3.10.11 in a local `.venv`.
- **Consequences:** Reproducible environment; isolated from system Python; slightly older
  interpreter than the machine default.

---

## D-04 — Install CUDA PyTorch before Anomalib

- **Date:** 2026-11-02
- **Status:** Accepted
- **Context:** System PyTorch was `2.10.0+cpu`. Anomalib depends on PyTorch and Lightning.
- **Alternatives:** Let Anomalib resolve PyTorch; install everything from PyPI.
- **Decision:** Install `torch`/`torchvision` from the CUDA 12.8 index first, then Anomalib,
  then re-verify CUDA.
- **Consequences:** GPU works (`2.11.0+cu128`, `cuda True`). Anomalib did not downgrade it,
  but the verification step remains mandatory for reproducibility.

---

## D-05 — Pivot from industrial images to surveillance video

- **Date:** 2026-11-02
- **Status:** Accepted (supersedes the original MVTec-AD plan)
- **Context:** The user explicitly wanted video. Verified that MVTec-AD and MVTec AD 2 are
  still-image datasets with no video clips. Anomalib supports `Avenue`, `ShanghaiTech`,
  `UCSDped` and video-native models.
- **Alternatives:** Stay on MVTec images; synthesize video; use a custom industrial clip.
- **Decision:** Video anomaly detection on surveillance benchmarks. The user accepted
  surveillance as the domain.
- **Consequences:** Loses the "industrial" theme; gains real video benchmarks, temporal
  analysis, and video-native models. The original PaDiM/PatchCore/WinCLIP comparison is
  retained, with AI-VAD/FUVAS added.

---

## D-06 — ShanghaiTech Campus as the primary dataset

- **Date:** 2026-11-02
- **Status:** Accepted
- **Context:** Need a standard video anomaly benchmark with localization labels.
- **Alternatives:** Avenue (small, 47 events), UCSD Ped (small), XD-Violence / UCF-Crime
  (large, weakly labeled).
- **Decision:** ShanghaiTech Campus: 13 scenes, ~330 normal training clips, ~107 test clips,
  130 abnormal events, frame and pixel masks.
- **Consequences:** More scenes and labels than Avenue; manual download required; heavier
  disk and preprocessing.

---

## D-07 — Avenue and UCSD Ped as secondary datasets

- **Date:** 2026-11-02
- **Status:** Accepted
- **Context:** Need a fast sanity-check dataset and a second benchmark.
- **Alternatives:** Only ShanghaiTech; add a large weakly supervised dataset.
- **Decision:** Avenue (small, 16/21 videos) and UCSD Ped1/Ped2 (frame folders).
- **Consequences:** Avenue enables quick iteration and cross-dataset comparison later.
  Adds storage and layout work but little risk.

---

## D-08 — Start with 3 scenes, then all 13

- **Date:** 2026-11-02
- **Status:** Accepted
- **Context:** Running all 13 scenes × 5 models with limited RAM would be slow.
- **Decision:** Scenes 1–3 for iteration; all 13 for the final benchmark.
- **Consequences:** Faster feedback; final table requires a longer run later.

---

## D-09 — Model lineup: PaDiM, PatchCore, AI-VAD, FUVAS, WinCLIP

- **Date:** 2026-11-02
- **Status:** Accepted
- **Context:** Need a progression story from classical to modern to foundation models.
- **Alternatives:** Only PaDiM + PatchCore + WinCLIP (original image plan); add EfficientAd.
- **Decision:** PaDiM (classical), PatchCore (modern frame-wise), AI-VAD (video-native),
  FUVAS (few-shot video), WinCLIP (zero-shot).
- **Consequences:** Covers frame-wise vs video-native and trained vs few-shot vs zero-shot.
  More models to run; AI-VAD/FUVAS may need extra weights and tuning.

---

## D-10 — Add WinCLIP last

- **Date:** 2026-11-02
- **Status:** Accepted
- **Context:** Zero-shot CLIP adds a separate dependency and prompt engineering.
- **Decision:** Finish baselines and the eval harness first; add WinCLIP after.
- **Consequences:** Lower risk of blocking progress on CLIP setup.

---

## D-11 — Gradio for the demo

- **Date:** 2026-11-02
- **Status:** Accepted
- **Context:** Need an upload-and-see-result interface for videos.
- **Alternatives:** Streamlit, CLI only.
- **Decision:** Gradio.
- **Consequences:** Fast video upload and annotated output; easy local hosting.

---

## D-12 — Frame-wise image models on video

- **Date:** 2026-11-02
- **Status:** Accepted
- **Context:** PaDiM/PatchCore expect images. Anomalib's video datamodule can yield clips.
- **Alternatives:** Skip image models; adapt models to clips.
- **Decision:** Use `clip_length_in_frames=1` for PaDiM/PatchCore/WinCLIP, producing normal
  `(B, 3, H, W)` batches; use clips only for AI-VAD/FUVAS.
- **Consequences:** Image models run unmodified; they remain blind to motion, which is
  analyzed explicitly in the report.

---

## D-13 — Disable the image visualizer for video data

- **Date:** 2026-11-02
- **Status:** Accepted
- **Context:** Anomalib's `ImageVisualizer` crashed:
  `'VideoItem' object has no attribute 'image_path'`.
- **Alternatives:** Patch the visualizer; suppress the callback; use the video visualizer.
- **Decision:** Pass `visualizer=False` when using image models on video data.
- **Consequences:** Tests complete and metrics are correct; visual heatmap rendering must be
  produced by our own plotting code.

---

## D-14 — Subsample normal frames for the PatchCore memory bank

- **Date:** 2026-11-02
- **Status:** Accepted
- **Context:** 69,976 frames × 1,024 patches × 1,536 dims × 4 bytes ≈ 440 GB for scene 1.
  System RAM is 7.9 GB.
- **Alternatives:** Full frames (impossible); lower resolution (helps but insufficient);
  feature dimension reduction; random subset.
- **Decision:** Use every 100th normal frame (~700 frames) for the first run. Make this
  configurable per scene.
- **Consequences:** Feasible memory and time. The memory bank may miss rare normal patterns;
  this must be reported and ablated (400 / 700 / 1,000 / 2,000 frames).

---

## D-15 — Subsample test frames for pixel metrics

- **Date:** 2026-11-02
- **Status:** Provisional
- **Context:** Full test (11,894 frames) ran inference but AUROC tried to allocate 6.2 GB
  and crashed on a 7.9 GB machine.
- **Alternatives:**
  1. Subsample test frames (current).
  2. Lower input resolution.
  3. Compute image AUROC on all frames and pixel AUROC on a subset.
  4. Stream per-video metrics without accumulating all pixel predictions.
- **Decision:** Use `frames_between_clips=10` (1,204 frames) for the first real number.
  Long-term, implement streaming/per-video evaluation.
- **Consequences:** Metrics are valid for the sampled subset but not the full benchmark.
  The report must state the protocol. Final results must improve this.

---

## D-16 — Use the Hugging Face mirror for ShanghaiTech

- **Date:** 2026-11-02
- **Status:** Accepted (Superseded the OneDrive route)
- **Context:** Official OneDrive folder share returned 403 to automated requests; Chrome
  download disappeared; FDM could not enumerate a folder share.
- **Alternatives:** Manual browser download; JDownloader 2; other mirrors.
- **Decision:** Download the 7-part archive from `123zdm/ShanghaiTech` on Hugging Face.
- **Consequences:** Reliable and fast. Integrity verified by expected byte sizes and a
  byte-exact join. The mirror is third-party, so cite the original dataset and license.

---

## D-17 — Accept Anomalib's XVID re-encode for ShanghaiTech

- **Date:** 2026-11-02
- **Status:** Accepted
- **Context:** Hardlinking raw videos caused
  `Requested next frame while there are no more frames left to decode`.
- **Alternatives:** Skip bad videos; patch the indexer; decode with OpenCV in a custom dataset.
- **Decision:** Remove hardlinks and let Anomalib re-encode all 330 training videos.
- **Consequences:** 330 extra files (~2.28 GB) and 15–25 minutes of one-time work, but stable
  frame indexing and training.

---

## D-18 — Store FFmpeg shared DLLs inside the TorchCodec package

- **Date:** 2026-11-02
- **Status:** Accepted
- **Context:** TorchCodec could not load FFmpeg. Adding to `PATH` worked only for that
  process; copying to `.venv/Scripts` did not work.
- **Alternatives:** Modify user/system PATH; copy DLLs into every working directory; use a
  launcher script.
- **Decision:** Copy the FFmpeg DLLs into `.venv/Lib/site-packages/torchcodec/`.
- **Consequences:** Works without environment changes. The fix is local to the venv and must
  be repeated if the venv is rebuilt.

---

## D-19 — Keep both the split parts and the joined archive (for now)

- **Date:** 2026-11-02
- **Status:** Provisional
- **Context:** Data occupies ~24.4 GB; the 7 parts and the joined tar.gz duplicate data.
- **Alternatives:** Delete parts after joining; delete joined archive after extraction.
- **Decision:** Keep them until the final extraction is confirmed and a report is written.
- **Consequences:** Uses ~13 GB extra. Can be cleaned later to free space.

---

## D-20 — Document the full journey in Markdown and PDF

- **Date:** 2026-11-02
- **Status:** Accepted
- **Context:** The user asked for a complete record of decisions, challenges, tuning, and
  the project journey, in MD and PDF.
- **Alternatives:** Only README; only a PDF; only chat history.
- **Decision:** Create `docs/JOURNEY.md`, `docs/DECISIONS.md`, `docs/CHALLENGES.md`,
  `docs/EXPERIMENTS.md`, and build `docs/Project_Journey.pdf` from them using Markdown →
  HTML (Python `markdown`) → Chrome headless print-to-PDF.
- **Consequences:** Reproducible documentation pipeline; the PDF can be regenerated after
  future experiments.

---

## Superseded / changed decisions

## D-21 — Harden AI-VAD as a track-level CCTV prototype

- **Date:** 2026-11-02
- **Status:** Accepted
- **Context:** Frame-level AI-VAD boxes can briefly appear on normal walking before
  returning to a fight. A product needs stable identities and alerts, not isolated boxes.
- **Decision:** Keep AI-VAD as the baseline, but align its input frame gap with deployment,
  log velocity/pose/deep components, associate detections with lightweight IoU tracks, and
  apply track-level persistence, hysteresis, debounce, cooldown, and held-out calibration.
- **Consequences:** The renderer becomes more suitable for CCTV iteration without discarding
  the current checkpoint. A tracker is intentionally lightweight for the prototype; ByteTrack
  or BoT-SORT can replace it later. AI-VAD remains an unusual-behavior detector, not a
  guaranteed fight classifier.

## D-22 - Use interval-scoped hard-negative mining

- **Date:** 2026-11-02
- **Status:** Accepted
- **Context:** ShanghaiTech test clips mix normal walking with labelled anomalies, so a
  whole clip cannot safely be added as normal calibration data.
- **Decision:** Use mask-zero, visually reviewed frame intervals for provisional hard-negative
  screening, and keep independent normal training footage for threshold calibration.
- **Consequences:** The workflow avoids label leakage but requires interval review before any
  refit. Camera-specific thresholds remain necessary.

## D-24 - Match AI-VAD training and inference gaps in real time

- **Date:** 2026-11-03
- **Status:** Accepted (supersedes the frame-count gap assumption in D-21)
- **Context:** The gap-5 demo produced boxes on ordinary walkers and scored only 18% of
  boxed frames on ground truth, while the gap-1 demo reached 94.5%. Investigation showed
  that training clips contained consecutive frames (one-frame gap), and that training
  videos are 24 fps while test frames are 10 fps.
- **Alternatives:** Keep tuning display thresholds at gap 5; switch to gap 1; fit a
  gap-matched checkpoint; resample all data to a common fps.
- **Decision:** Fit AI-VAD with an exact within-clip frame step chosen to match the
  inference gap in seconds. For scene 1, use a 5-frame step at 24 fps (0.208 s) with a
  2-frame inference gap at 10 fps (0.2 s). Keep gap-1 rendering only for comparisons.
- **Consequences:** New checkpoints per temporal setting; the custom indexer lives in
  `src/fit_aivad_gap.py`. Gap-5 inference with the old gap-1 checkpoint is retired.

## D-25 - Use pixel masks for per-person evaluation and a focus lock for display

- **Date:** 2026-11-03
- **Status:** Accepted
- **Context:** Frame-level precision hid the real problem: a frame could be correctly
  marked anomalous while the box sat on a walker. The dataset provides pixel masks, and
  the renderer logs every person candidate.
- **Alternatives:** Continue tuning frame-level thresholds; manually label person boxes;
  train a supervised verifier first.
- **Decision:** Use the pixel masks to label person candidates automatically for
  **evaluation only** (box at least 25% covered by the mask, or containing at least 40% of
  it). For display, add a temporal focus lock that prefers candidates near the current
  event and rejects large jumps, plus a 90 px anchor radius so both fighters are boxed.
- **Consequences:** Honest per-person numbers (0.888 precision / 0.812 recall on
  `01_0026`) and a clean demo. Thresholds and focus radius are tuned on this clip, so the
  numbers are an upper bound. Pixel masks are never used as model input. A supervised
  verifier remains the next step for cross-camera reliability.

## D-23 - Do not claim fight detection from the current AI-VAD demo

- **Date:** 2026-11-03
- **Status:** Accepted
- **Context:** The gap-5 fight demo improved visually, but a second clip (`03_0061`)
  produced many boxes on ordinary walking.
- **Decision:** Describe AI-VAD as an unusual-human-behavior prototype until walking
  hard-negative refitting and component ablation demonstrate reliable event separation.
- **Consequences:** The current videos are evidence of pipeline behavior and failure modes,
  not product-level fight-detection validation.

| Original | Replaced by | Reason |
|---|---|---|
| Industrial image anomaly detection on MVTec-AD | Video anomaly detection on ShanghaiTech/Avenue/UCSD | User wanted video; MVTec has no video |
| 15 MVTec categories | 13 ShanghaiTech scenes | Dataset pivot |
| PatchCore + WinCLIP only | PaDiM, PatchCore, AI-VAD, FUVAS, WinCLIP | Video-native and few-shot methods added |
| OneDrive browser download | Hugging Face mirror | 403 / unreliable download |
| Python 3.13 | Python 3.10 venv | Anomalib tested range |
| CPU PyTorch | CUDA 12.8 PyTorch | GPU acceleration |
