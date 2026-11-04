# Video Anomaly Detection — Literature Review

Scope: unsupervised / one-class, weakly-supervised, and zero-shot video anomaly
detection (VAD), with emphasis on the lineage this project already touches
(frame-wise feature embedding, video-native appearance+motion, CLIP / foundation
models), plus the evaluation-protocol literature.

---

## 1. Classical / hand-crafted era (2009–2013)
Define the problem, the datasets, and the evaluation metrics (frame/pixel AUROC,
localization).

- Kim & Grauman, *Observe Locally, Infer Globally: A Space-Time MRF for Detecting
  Abnormal Activities with Incremental Updates*, CVPR 2009.
- Mahadevan, Li, Bhalodia, Vasconcelos, *Anomaly Detection in Crowded Scenes*, CVPR
  2010 (MDT).
- Cong, Yuan, Liu, *Sparse Reconstruction Cost for Abnormal Event Detection*, CVPR 2011.
- Zhao, Fei-Fei, Xing, *Online Detection of Unusual Events in Videos via Dynamic Sparse
  Coding*, CVPR 2011.
- Lu, Shi, Jia, *Abnormal Event Detection at 150 FPS in MATLAB*, ICCV 2013 (Avenue
  dataset; the sparse-combination baseline).

## 2. Deep reconstruction / prediction (2018–2020)
The generation that made ShanghaiTech a standard benchmark.

- Liu, Luo, Lian, Gao, *Future Frame Prediction for Anomaly Detection — A New Baseline*,
  CVPR 2018 (**introduces ShanghaiTech**).
- Gong et al., *Memorizing Normality to Detect Anomaly: Memory-augmented Deep
  Autoencoder (MemAE)*, ICCV 2019.
- Park, Noh, Ham, *Learning Memory-Guided Normality for Anomaly Detection (MNAD)*,
  CVPR 2020.
- Georgescu et al., *A Background-Agnostic Framework with Adversarial Training for
  Abnormal Event Detection in Video*, TPAMI 2021.
- Georgescu et al., *Anomaly Detection in Video via Self-Supervised and Multi-Task
  Learning*, CVPR 2021.
- Liu et al., *Clustering Driven Deep Autoencoder for Video Anomaly Detection*, ECCV 2020.

## 3. Frame-wise feature-embedding / density (image-AD lineage, reused on video)
These are *image* anomaly-detection methods adapted to video; they are motion-blind —
which this project demonstrated empirically (PatchCore misses the fight).

- Defard, Setkov, Loesch, Audigier, *PaDiM: a Patch Distribution Modeling Framework for
  Anomaly Detection and Localization*, ICPR-W 2020.
- Roth et al., *Towards Total Recall in Industrial Anomaly Detection (PatchCore)*,
  CVPR 2022.
- Yu et al., *FastFlow: Unsupervised Anomaly Detection and Localization via 2D
  Normalizing Flows*, 2021.
- Gudovskiy et al., *CFLOW-AD: Real-Time Unsupervised Anomaly Detection with
  Localization via Conditional Normalizing Flows*, WACV 2022.

## 4. Video-native appearance + motion (the AI-VAD lineage) — home turf
- Reiss & Hoshen, *Mean-Shifted Contrastive Loss for Anomaly Detection (MSCL)*, AAAI 2023.
- Reiss & Hoshen, *AI-VAD: Accurate and Interpretable Video Anomaly Detection*, CVPR
  2024 (the model this project uses).
- Reiss & Hoshen, *Attribute-based Representations for Accurate and Interpretable Video
  Anomaly Detection (ARVAD)*, CVPR 2024.
- Georgescu et al., *Self-Supervised Attentive Generative Adversarial Networks for
  Video Anomaly Detection (SSAGAN)*, IEEE TNNLS 2022.
- FUVAS, *Few-shot Unsupervised Video Anomaly Segmentation* (Anomalib's few-shot video
  method).

## 5. Weakly-supervised VAD (MIL, top-down) — where SOTA lives
- Sultani, Chen, Shah, *Real-world Anomaly Detection in Surveillance Videos*, CVPR 2018
  (**introduces UCF-Crime**; MIL ranking).
- Zhong et al., *Graph Convolutional Label Noise Cleaner*, CVPR 2019.
- Tian et al., *Weakly-supervised Video Anomaly Detection with Robust Temporal Feature
  Magnitude Learning (RTFM)*, ICCV 2021.
- Wu et al., *Not Only Look, But Also Listen*, CVPR 2020 (**introduces XD-Violence**).
- Wan et al., *Weakly Supervised Video Anomaly Detection via Center-guided Discriminative
  Learning*, ICME 2020.
- Feng et al., *Learning Multi-Granularity Discriminative Representations (MIST)*,
  NeurIPS 2021.

## 6. CLIP / foundation-model / zero-shot VAD (2023–2025) — the frontier
- Jeong et al., *WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation*, CVPR
  2023.
- Zhou et al., *AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly
  Detection*, ICLR 2024.
- Wu et al., *VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video
  Anomaly Detection*, ICLR 2024.
- HOLMES and related VLM-reasoning anomaly detectors, 2024.
- AnomalyGPT / MMAD: multimodal LLM-based VAD, 2024.
- Diffusion-based VAD: AnoDDPM (2022) and successors.

## 7. Efficient / online / streaming VAD
Relevant to this project's 7.9 GB RAM constraint.

- Real-time / streaming VAD formulations (e.g. RAD, CVPR 2024).
- Memory-bank compression: PatchCore coreset; FastFlow's real-time claim.

## 8. Evaluation-protocol literature (directly relevant to this project's critiques)
- Ramachandra, Jones, Vatsavai, *A Survey of Single-Scene Video Anomaly Detection*,
  TPAMI 2020.
- Georgescu et al. (TPAMI 2021) — argues AUROC is misleading; reports AP and
  object-centric localization (the citation for why a 0.989 pixel AUROC is suspect).
- Kiran, Thomas, Parakkal, *An Overview of Deep Learning Based Methods for
  Unsupervised and Semi-supervised Anomaly Detection in Videos*, J. Imaging 2018.
- Pang, Shen, Cao, van den Hengel, *Deep Learning for Anomaly Detection: A Review*,
  ACM Computing Surveys 2021.
- Per-Region Overlap (PRO) — localization metric adopted for VAD pixel evaluation.

## 9. Datasets
ShanghaiTech (2018), Avenue (2013), UCSD Ped1/Ped2 (2010), UCF-Crime (2018),
XD-Violence (2020), UBnormal (2022, synthetic for OOD generalization).

## 10. Few-shot, zero-shot, and meta-learning — project note

Meta-learning is a **few-shot** tool, not a zero-shot one. It learns a prior over tasks,
each with a small support set, then adapts at test time. Zero-shot has no support set, so
it needs semantic side information; in VAD that is the CLIP/foundation-model lineage
(§6), not classical meta-learning.

Relevant building blocks:

- **Metric-based:** Vinyals et al., *Matching Networks for One Shot Learning*, NeurIPS
  2016; Snell et al., *Prototypical Networks for Few-shot Learning*, NeurIPS 2017;
  Sung et al., *Learning to Compare: Relation Network for Few-Shot Learning*, CVPR 2018.
- **Optimization-based:** Finn et al., *Model-Agnostic Meta-Learning (MAML)*, ICML 2017;
  Nichol et al., *On First-Order Meta-Learning Algorithms (Reptile)*, 2018.
- **Few-shot AD in this project:** FUVAS (§4) is the existing few-shot baseline; a
  meta-learned normal prototype or calibration head should be compared at k = 1/2/4/8.
- **Zero-shot AD in this project:** WinCLIP/AnomalyCLIP (§6), not meta-learning.

The realistic use here is **meta-calibration**: treat each clip/camera as a task, use k
normal clips as support, and learn a threshold or monotone score transform that transfers
to query clips. This directly targets the cross-clip threshold failure in §11. Task design
is the hard part: this is one-class few-shot (support is normal-only), not N-way K-shot
classification.

## 11. Gap analysis — where this project sits

| Axis | This project | Frontier |
|---|---|---|
| Method | PatchCore (image) + AI-VAD (video) | CLIP/foundation, ARVAD, weakly-supervised |
| Metrics | subsampled AUROC, per-person pixel-mask precision | full-set AUROC/AP/PRO, multi-seed |
| Robustness | fixed-Δt invariance verified 10–30 fps | genuine high-fps data, cross-camera calibration |
| Novelty | frame-rate parameterisation + calibration study | frame-rate robustness / localization are open |

Three empirical findings are genuine, under-explored entry points:

1. **Train/inference frame-rate & gap mismatch** silently breaks flow-based video AD
   (24 fps training vs 10 fps test ⇒ 12× out of distribution for AI-VAD's velocity
   term). → RQ-1 in the roadmap. Phase 1 verified **fixed-Δt invariance** across
   10–30 fps; the remaining question is the value of smaller Δt at genuinely high fps.
2. **Frame-level AUROC hides per-person localization error** (a frame can be scored
   "anomalous" while the box sits on the wrong person). → RQ-2.
3. **Thresholds do not transfer across unseen clips/cameras.** A blind q80 threshold on
   Avenue `01` gave box precision 0.286; per-clip tuning reached 0.546 precision / 0.490
   recall. This is now C-37/C-38 and the strongest near-term research target.

All three are defensible, method-driven research questions. See `../RESEARCH_ROADMAP.md`.
