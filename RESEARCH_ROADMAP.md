# Research Roadmap: Video Anomaly Detection → Q1

> Companion to `literature review/`. This file is the execution plan that takes the
> current project from "a subsampled baseline + a hand-tuned demo" to a publishable
> first-author contribution in a top-tier venue.

---

## 0. Where we are (honest baseline)

| Asset | Status |
|---|---|
| PatchCore scene-1 result | Subsampled (every 100th train frame, every 10th test frame), not comparable to published numbers |
| Pixel AUROC 0.989 | Untrustworthy until recomputed streaming on the full test set |
| AI-VAD people-only demo | 98/98 boxed frames on `01_0026`, but tuned on that clip; `03_0061` still fails |
| Evaluation harness | No streaming metrics, no seeds, no multi-run variance, validation split has no anomalies |
| Other models | PaDiM / FUVAS / WinCLIP not benchmarked |

**Conclusion:** the *pipeline and documentation* are strong; the *results* are not yet
science. Everything below converts "a project" into "a paper".

---

## 1. Define "Q1" realistically

Target trajectory (first → stretch):

1. **First submission:** WACV / ACCV / BMVC, or a top-tier workshop (e.g. CVPR-W on
   anomaly detection / safety).
2. **Main goal:** CVPR / ICCV / ECCV.
3. **Journal:** TIP / TMM / PR (solid), **TPAMI / IJCV** (stretch).

A Q1 paper must have all four of:

- a **clear, falsifiable research question**,
- **SOTA-comparable, reproducible numbers** on standard benchmarks with standard metrics,
- a **novel component** whose contribution is isolated by ablation,
- **error analysis** that shows you understand *where and why* the method works.

---

## 2. Pick the research question (do this first — it decides everything)

Ranked by fit with what you already have:

### RQ-1 (recommended) — Frame-rate & temporal-calibration robust video AD
- **Your finding:** AI-VAD's velocity term silently breaks when the train/inference
  frame gap or fps mismatch (24 fps train vs 10 fps test ⇒ 12× out of distribution).
- **Contribution:** formalize the failure, propose a *frame-rate-invariant motion
  feature* and/or a *temporal-calibration module*, ablate across 10/24/30 fps and
  across datasets.
- **Why it wins:** it is a real, under-explored methodological gap; you already have
  the evidence; it generalizes to every flow-based video AD method.

### RQ-2 — Per-person localization: metrics + person-centric VAD
- **Your finding:** frame-level AUROC hides the wrong-box problem (a frame can be
  "correct" while the box sits on a walker).
- **Contribution:** (a) a person-level localization metric derived from pixel masks,
  (b) a person-centric / interaction-aware scoring head that improves it.
- **Why it wins:** directly reuses your pixel-mask evaluation work; addresses a real
  criticism of the field.

### RQ-3 — Memory-efficient video-native AD
- **Your constraint:** 7.9 GB RAM forced heavy subsampling.
- **Contribution:** a bank-compressed / streaming AI-VAD that matches SOTA at a
  fraction of memory.
- **Why it's last:** competitive and crowded; "efficient AD" is a busy lane.

**Decision gate:** run a 2-week feasibility spike on RQ-1 before committing. If the
frame-rate failure reproduces on a second method (e.g. MNAD or ARVAD), commit to RQ-1.

## 2b. Current focus — higher-frame-rate detection

Standard CCTV runs at **25/30 fps**, high-end cameras at **50/60 fps**. The model is
time-based (`Δt ≈ 0.2 s`), so it can run on any source *today* by choosing the gap that
preserves the real-time window: `gap = round(source_fps × Δt)`.

| Source fps | Frame gap for Δt = 0.2 s |
|---:|---:|
| 10 | 2 |
| 15 | 3 |
| 24 / 25 | 5 |
| 30 | 6 |
| 50 | 10 |
| 60 | 12 |

Two questions drive the next phase:

- **Invariance (engineering):** does the detector behave identically at a fixed Δt across
  10/25/30/60 fps? If yes, the model is temporally calibrated, not fps-dependent.
- **Value of resolution (research):** does a smaller Δt at high fps improve fast-motion
  detection (fights), or only add compute?

Steps:

1. Make the sampler fps-aware (`--source-fps` + `--target-dt` instead of `--frame-gap`).
2. Build a multi-fps labelled evaluation set (same scene at 10/25/30/60 fps).
3. Run the fixed-Δt invariance ablation (velocity distribution and AUC vs fps).
4. Refit at smaller Δt (0.05 / 0.1 / 0.2 s) with `src/fit_aivad_gap.py` and compare.

> **Status 2026-11-03:** steps 1 and 3 are done (`docs/FPS_ROBUSTNESS.md`). At
> fixed Δt = 0.2083 s, 20/30 fps emulations match the 10 fps run (Spearman 0.999,
> alarm Jaccard 0.987–0.994, zero delay); 5 fps keeps score/alarm agreement but loses
> box localization. A real 25→10 fps Avenue comparison keeps ρ = 0.987. Remaining:
> the matched Δt sweep on Avenue and genuine high-fps data (50/60 fps out of scope).

---

## 3. Phase plan

### Phase 1 — Benchmark harness (2–3 weeks)
- Streaming / per-video metrics: frame AUROC + AP, pixel AUROC + PRO, no OOM.
- Fixed seeds; ≥3 runs; report mean ± std.
- Standard protocol on **ShanghaiTech (all 13 scenes), Avenue, UCSD Ped1/Ped2**,
  then extend to **UCF-Crime / XD-Violence**.
- Fix the validation split so F1 thresholds are legitimate.

**Exit criteria:** full-set PatchCore/PaDiM numbers that a reviewer would accept.

### Phase 2 — Reproduce the field (4–6 weeks)
Baselines at matched protocol: PaDiM, PatchCore, MNAD, AI-VAD, ARVAD, VadCLIP,
WinCLIP. This is the table every reviewer expects, and it tells you *where* to win.

### Phase 3 — Novel component (4–6 weeks)
- One clear delta over the nearest baseline (AI-VAD or ARVAD).
- Isolate it in a controlled ablation.
- **Remove or formalize** every demo heuristic (focus lock, cluster radius, percentile
  normalization): each must become a principled, ablated module or leave the paper.

### Phase 4 — Error analysis (2 weeks)
Per-event recall, detection-delay distribution, false-alarm rate, cross-camera
failure cases. This is the difference between a leaderboard grab and a Q1 paper.

### Phase 5 — Write + release (3–4 weeks)
- Public repo: configs, seeds, checkpoints, and the benchmark harness (a service to the
  community).
- Intro written around the *insight*, not the pipeline.

---

## 4. Compute plan

| Hardware | Suits | Not suited |
|---|---|---|
| RTX 3060 12 GB | PaDiM, PatchCore, AI-VAD, small ablate runs | MNAD, VadCLIP, diffusion, 13-scene sweeps |
| Rent multi-GPU (Lambda/Vast.ai/Colab Pro) | full sweeps, CLIP baselines, UCF-Crime/XD-Violence | — |

Budget ~$100–300 of cloud GPU for the full benchmark + baselines.

---

## 5. Key risks

- **Insight not novel / already published.** Mitigate with a literature sweep before
  committing (see `literature review/`).
- **Compute ceiling** on a single 3060. Mitigate with cloud and low-res experiments
  for the expensive baselines.
- **Overfitting the demo again.** Rule: *never tune on the clip you evaluate.*
- **Scope creep.** One RQ, one dataset set, one venue. Ship.

---

## 6. Milestones (target 6–9 months)

| Month | Milestone |
|---|---|
| 1 | Harness done; full-set PatchCore/PaDiM + seeds |
| 2 | Baselines reproduced; feasibility spike on RQ-1 |
| 3 | Novel component working; first ablations |
| 4–5 | Full benchmark (13 scenes + UCF-Crime/XD-Violence) |
| 6 | Error analysis + write-up + code release |
| 7–8 | Submit, rebuttal |
| 9 | Resubmit / journal |

---

## 7. Definition of done (Q1)

- Reproducible results table, mean±std, on ≥2 standard benchmarks.
- Novel component ablated and shown to beat the nearest baseline by a clear margin.
- Frame-rate / localization insight demonstrated with controlled experiments.
- Public code + the benchmark harness.
- Error analysis section that anticipates and answers reviewer objections.
