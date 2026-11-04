"""Shared AI-VAD inference, scoring, selection, and tracking code.

This module is the single implementation used by the renderer, the headless
evaluator, and the calibration tools.  It wraps Anomalib's AI-VAD submodules so the
project can log component scores and apply its own temporal selection rules without
duplicating the model plumbing in every script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from anomalib.models import AiVad
from anomalib.models.video.ai_vad.features import FeatureType

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = (
    ROOT
    / "results/aivad/scene01/AiVad/ShanghaiTech/v0/weights/lightning/model.ckpt"
)
ROBUST_GAP5_CKPT = (
    ROOT
    / "results/aivad/scene01_gap5/AiVad/ShanghaiTech/v0/weights/lightning/model_robust.ckpt"
)

DEFAULT_WEIGHTS = {"velocity": 0.6, "pose": 0.25, "deep": 0.15}


@dataclass
class InferenceResult:
    """Per-pair AI-VAD output."""

    score: float
    anomaly_map: np.ndarray
    region: dict | None
    region_scores: torch.Tensor | None
    component_scores: dict[str, torch.Tensor]


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    score: float
    track_id: int | None = None

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


@dataclass
class Track:
    track_id: int
    detection: Detection
    scores: list[float]
    missed: int = 0
    cooldown: int = 0
    active: bool = False


def load_model(checkpoint: Path | str, device: str) -> AiVad:
    """Load an AI-VAD checkpoint for inference only."""
    model = AiVad.load_from_checkpoint(
        str(checkpoint), map_location=device, post_processor=False
    )
    model.model.eval().to(device)
    return model


def to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def component_weights(
    score_mode: str = "weighted",
    velocity_weight: float = 0.6,
    pose_weight: float = 0.25,
    deep_weight: float = 0.15,
) -> dict[str, float]:
    """Return the component weights for ``mode``."""
    if score_mode == "sum":
        return {"velocity": 1.0, "pose": 1.0, "deep": 1.0}
    return {
        "velocity": velocity_weight,
        "pose": pose_weight,
        "deep": deep_weight,
    }


def _empty_result(height: int, width: int) -> InferenceResult:
    # Legacy behaviour: a pair with no person regions scores 1.0 (below the 1.4
    # display threshold used by the demo) and produces an empty map.
    return InferenceResult(
        score=1.0,
        anomaly_map=np.zeros((height, width), dtype=np.float32),
        region=None,
        region_scores=None,
        component_scores={},
    )


def _predict_components(
    model: AiVad,
    feature_set: dict,
    weights: dict[str, float],
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    estimator = model.model.density_estimator
    component_scores: dict[str, torch.Tensor] = {}
    if estimator.use_velocity_features and feature_set.get(FeatureType.VELOCITY) is not None:
        values = feature_set[FeatureType.VELOCITY]
        if values.numel():
            component_scores["velocity"] = estimator.velocity_estimator.predict(values)
    if estimator.use_deep_features and feature_set.get(FeatureType.DEEP) is not None:
        values = feature_set[FeatureType.DEEP]
        if values.numel():
            component_scores["deep"] = estimator.appearance_estimator.predict(values)
    if estimator.use_pose_features and feature_set.get(FeatureType.POSE) is not None:
        values = feature_set[FeatureType.POSE]
        if values.numel():
            component_scores["pose"] = estimator.pose_estimator.predict(values)
    if not component_scores:
        return {}, None
    region_scores = sum(
        weights.get(name, 0.0) * values for name, values in component_scores.items()
    )
    return component_scores, region_scores


def infer_batch(
    model: AiVad,
    first_frames: list[np.ndarray],
    last_frames: list[np.ndarray],
    device: str,
    weights: dict[str, float] | None = None,
) -> list[InferenceResult]:
    """Run AI-VAD on a batch of frame pairs.

    Batching RAFT, Mask R-CNN, and the feature extractors is the main speed-up for
    the frame-rate sweep.  Items without person regions are returned as empty
    results instead of failing the batch.
    """
    if len(first_frames) != len(last_frames):
        raise ValueError("first_frames and last_frames must have the same length")
    if not first_frames:
        return []
    weights = weights or DEFAULT_WEIGHTS
    height, width = last_frames[0].shape[:2]

    clip = torch.stack(
        [
            torch.stack([to_tensor(first), to_tensor(last)])
            for first, last in zip(first_frames, last_frames, strict=True)
        ]
    ).to(device)

    results: list[InferenceResult] = [
        _empty_result(height, width) for _ in range(len(first_frames))
    ]
    with torch.no_grad():
        first = clip[:, 0]
        last = clip[:, -1]
        flow = model.model.flow_extractor(first, last)
        regions = model.model.region_extractor(first, last)

        nonempty = [
            index
            for index, region in enumerate(regions)
            if "boxes" in region and len(region["boxes"]) > 0
        ]
        if not nonempty:
            return results

        # Anomalib's AI-VAD forward pass extracts appearance/pose features from the
        # first frame and detects regions on the last frame.  Preserve that order.
        rgb_subset = first[nonempty]
        flow_subset = flow[nonempty]
        regions_subset = [regions[index] for index in nonempty]

        try:
            features = model.model.feature_extractor(
                rgb_subset, flow_subset, regions_subset
            )
        except Exception:
            # A single malformed region can abort the whole batch; retry one by one
            # and keep the successful items.
            features = []
            for offset in range(len(regions_subset)):
                try:
                    item = model.model.feature_extractor(
                        rgb_subset[offset : offset + 1],
                        flow_subset[offset : offset + 1],
                        [regions_subset[offset]],
                    )[0]
                except Exception:
                    item = None
                features.append(item)

        for local_index, batch_index in enumerate(nonempty):
            feature_set = features[local_index]
            if feature_set is None:
                continue
            component_scores, region_scores = _predict_components(
                model, feature_set, weights
            )
            if region_scores is None or region_scores.numel() == 0:
                continue
            region = regions[batch_index]
            anomaly_map = torch.amax(
                region["masks"] * region_scores.view(-1, 1, 1, 1), dim=0
            )
            image_score = sum(
                weights.get(name, 0.0) * values.max()
                for name, values in component_scores.items()
            )
            results[batch_index] = InferenceResult(
                score=float(image_score.detach().cpu()),
                anomaly_map=anomaly_map.squeeze().detach().float().cpu().numpy(),
                region={"boxes": region["boxes"].detach().cpu()},
                region_scores=region_scores.detach().float().cpu(),
                component_scores={
                    name: values.detach().float().cpu()
                    for name, values in component_scores.items()
                },
            )
    return results


def infer_pair(
    model: AiVad,
    first_bgr: np.ndarray,
    last_bgr: np.ndarray,
    device: str,
    weights: dict[str, float] | None = None,
) -> tuple[float, np.ndarray, dict | None, torch.Tensor | None, dict[str, torch.Tensor]]:
    """Single-pair wrapper kept for compatibility with the existing scripts."""
    result = infer_batch(model, [first_bgr], [last_bgr], device, weights)[0]
    return (
        result.score,
        result.anomaly_map,
        result.region,
        result.region_scores,
        result.component_scores,
    )


def score_persons(
    anomaly_map: np.ndarray,
    region: dict | None,
    region_scores: torch.Tensor | None,
    component_scores: dict[str, torch.Tensor],
    min_area: int,
) -> list[dict]:
    """Score every detected person with the weighted and per-component scores."""
    if region is None or "boxes" not in region or len(region["boxes"]) == 0:
        return []

    height, width = anomaly_map.shape
    boxes = region["boxes"].cpu().numpy()
    weighted = (
        region_scores.numpy()
        if region_scores is not None and len(region_scores) == len(boxes)
        else None
    )
    components = {
        name: values.numpy()
        for name, values in component_scores.items()
        if len(values) == len(boxes)
    }

    candidates = []
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = (int(value) for value in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        area = (x2 - x1) * (y2 - y1)
        if area < min_area:
            continue
        score = (
            float(weighted[index])
            if weighted is not None
            else float(anomaly_map[y1:y2, x1:x2].max())
        )
        candidates.append(
            {
                "x": x1,
                "y": y1,
                "w": x2 - x1,
                "h": y2 - y1,
                "area": area,
                "score": score,
                "velocity": float(components["velocity"][index])
                if "velocity" in components
                else 0.0,
                "pose": float(components["pose"][index])
                if "pose" in components
                else 0.0,
                "deep": float(components["deep"][index])
                if "deep" in components
                else 0.0,
            }
        )
    return candidates


def select_persons(
    candidates: list[dict],
    threshold: float,
    relative_threshold: float,
    motion_threshold: float,
    cluster_radius: float,
    top_k: int,
    focus_center: tuple[float, float] | None = None,
    focus_radius: float = 0.0,
    focus_ratio: float = 0.8,
    focus_jump: float = 1.6,
    require_motion: bool = True,
) -> tuple[list[Detection], tuple[float, float] | None]:
    """Pick the people that deserve a box and return the new focus point."""
    if not candidates:
        return [], focus_center

    if require_motion:
        candidates = [
            candidate
            for candidate in candidates
            if candidate["velocity"] >= motion_threshold
        ]
    if not candidates:
        return [], focus_center

    top_score = max(candidate["score"] for candidate in candidates)
    top = max(candidates, key=lambda candidate: candidate["score"])

    if focus_center is not None and focus_radius > 0:
        top_cx = top["x"] + top["w"] / 2
        top_cy = top["y"] + top["h"] / 2
        top_distance = (
            (top_cx - focus_center[0]) ** 2 + (top_cy - focus_center[1]) ** 2
        ) ** 0.5

        near = []
        for candidate in candidates:
            cx = candidate["x"] + candidate["w"] / 2
            cy = candidate["y"] + candidate["h"] / 2
            distance = ((cx - focus_center[0]) ** 2 + (cy - focus_center[1]) ** 2) ** 0.5
            if distance <= focus_radius and candidate["score"] >= focus_ratio * top_score:
                near.append(candidate)
        if near:
            top = max(near, key=lambda candidate: candidate["score"])
        elif top_distance > focus_radius * focus_jump:
            # The only strong candidate jumped away from the tracked event.
            return [], focus_center

    cutoff = max(threshold, relative_threshold * top["score"])
    top_cx = top["x"] + top["w"] / 2
    top_cy = top["y"] + top["h"] / 2

    selected = []
    for candidate in candidates:
        if candidate["score"] < cutoff:
            continue
        if cluster_radius > 0:
            cx = candidate["x"] + candidate["w"] / 2
            cy = candidate["y"] + candidate["h"] / 2
            distance = ((cx - top_cx) ** 2 + (cy - top_cy) ** 2) ** 0.5
            if distance > cluster_radius and candidate is not top:
                continue
        selected.append(candidate)

    selected.sort(key=lambda candidate: (-candidate["score"], -candidate["area"]))
    new_focus = (top_cx, top_cy) if selected else focus_center
    return (
        [
            Detection(x=c["x"], y=c["y"], w=c["w"], h=c["h"], score=c["score"])
            for c in selected[:top_k]
        ],
        new_focus,
    )


def iou(left: Detection, right: Detection) -> float:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x + left.w, right.x + right.w)
    y2 = min(left.y + left.h, right.y + right.h)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = left.w * left.h + right.w * right.h - intersection
    return intersection / union if union else 0.0


def update_tracks(
    tracks: dict[int, Track],
    detections: list[Detection],
    next_id: int,
    iou_threshold: float,
    window: int,
    min_hits: int,
    stop_hits: int,
    cooldown_frames: int,
    detection_threshold: float,
    stop_threshold: float,
) -> tuple[dict[int, Track], int, list[Detection], int]:
    unmatched_tracks = set(tracks)
    unmatched_detections = set(range(len(detections)))
    matches = []
    for track_id, track in tracks.items():
        best = max(
            (
                (iou(track.detection, detection), index)
                for index, detection in enumerate(detections)
                if index in unmatched_detections
            ),
            default=(0.0, -1),
        )
        if best[0] >= iou_threshold:
            matches.append((track_id, best[1]))
            unmatched_tracks.discard(track_id)
            unmatched_detections.discard(best[1])

    for track_id, detection_index in matches:
        track = tracks[track_id]
        detection = detections[detection_index]
        track.detection = detection
        track.scores = (track.scores + [detection.score])[-window:]
        track.missed = 0
        if track.cooldown:
            track.cooldown -= 1
        detection.track_id = track_id

    for track_id in unmatched_tracks:
        tracks[track_id].missed += 1
        if tracks[track_id].cooldown:
            tracks[track_id].cooldown -= 1
    for detection_index in unmatched_detections:
        detection = detections[detection_index]
        tracks[next_id] = Track(next_id, detection, [detection.score])
        detection.track_id = next_id
        next_id += 1

    for track_id in list(tracks):
        if tracks[track_id].missed > stop_hits:
            del tracks[track_id]

    visible = []
    for track in tracks.values():
        if track.missed:
            continue
        track_score = float(np.median(track.scores))
        hits = sum(score >= detection_threshold for score in track.scores)
        if hits >= min_hits and track_score >= detection_threshold:
            track.active = True
        elif track.active and track_score < stop_threshold:
            track.active = False
        elif len(track.scores) >= window and hits == 0:
            track.active = False
        if track.missed > cooldown_frames:
            track.active = False
        if track.active and track.cooldown == 0:
            visible.append(track.detection)
    return tracks, next_id, visible, len(matches)
