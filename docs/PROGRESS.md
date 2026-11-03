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

## Current state

- Environment: ready.
- Data: ready (all three datasets).
- Code: PatchCore and AI-VAD people-only renderers are implemented; the general runner,
  Gradio app, and benchmark harness remain pending.
- Results: one PatchCore scene-1 benchmark result plus a corrected AI-VAD scene-1 demo.
- Next: implement the runner, then PaDiM scene 1 and the 3-scene comparison.

### Product-hardening pass

- AI-VAD now supports configurable temporal frame gaps instead of assuming consecutive
  frames.
- The renderer logs image, velocity, pose, deep, per-region, track-match, and active-track
  values in its CSV.
- Lightweight IoU tracks, rolling score histories, persistence, and hysteresis are wired in.
- `src/calibrate_people_thresholds.py` produces camera-specific thresholds from held-out
  normal frames.
- Initial gap 2/5/10 calibration is complete on independent scene-2/3 normal training
  footage; gap 5 produced image/region thresholds of 2.937/2.253 at the 99.5th percentile.
- A cleaned gap-5 fight render is available with persistence and stricter thresholds:
  `results/videos/shanghaitech_scene01_01_0026_people_gap5_clean.mp4`.
- A second validation render, `03_0061`, produced 131 boxed frames, mostly during
  ordinary walking (frames 76-174), confirming the false-positive issue generalizes.
- A provisional hard-negative screen over reviewed mask-zero intervals found higher
  walking/crowd maxima (4.288 image, 3.050 region at gap 5).
- The issue is diagnosed but not solved: AI-VAD pose/deep scores are high for walking,
  and tracking only stabilizes those detections. Pending: fit deployment-like gap
  checkpoints, finish interval review, refit with confirmed hard negatives, and report
  false alerts per camera-day, event recall, localization, latency, and FPS.

## Immediate next actions

1. [ ] Create `requirements.txt` from the verified environment.
2. [ ] Implement `src/run.py` (config-driven datamodule + model + engine).
3. [ ] Implement frame subsampling with per-scene target frames (avoid RAM blowup).
4. [ ] Implement memory-safe evaluation (image AUROC full test, pixel AUROC on a sample,
   or streaming per-video metrics).
5. [ ] Save/load checkpoints so evaluation does not require refitting.
6. [ ] Run PaDiM on scene 1 with the same protocol.
7. [ ] Run PaDiM + PatchCore on scenes 1-3 and build `results/summary.csv`.
8. [~] Run AI-VAD on scenes 1-3 (scene 1 fit/demo done; scenes 2–3 pending).
9. [ ] Add temporal smoothing + event metrics.
10. [ ] FUVAS few-shot, then WinCLIP zero-shot.
11. [ ] Gradio demo, ONNX export, FPS table.
12. [ ] Final report and README results table.

## Open questions

- Best memory-safe evaluation strategy: test-frame stride vs per-video streaming AUROC.
- Whether to keep full `samples` for image-level AUROC and subsample only for pixel AUROC.
- Optimal memory-bank size per scene (400 / 700 / 1,000 frames) for the AUROC/memory tradeoff.
- Whether AI-VAD/FUVAS need separate clip lengths and datamodule settings beyond defaults.
- Whether the UCSD `UCSDped` datamodule accepts our extracted folder layout directly.

## Risks being watched

- System RAM (7.9 GB) limiting full pixel-level evaluation.
- DataLoader speed with `num_workers=0`.
- Anomalib visualizer incompatibility with video batches (use `visualizer=False`).
- FUVAS/WinCLIP VRAM and model-weight downloads on a 12 GB card.
- Keeping the report honest about subsampled training/evaluation protocols.
