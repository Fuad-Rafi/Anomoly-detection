"""Render a people-only anomaly video with AI-VAD.

AI-VAD is fitted with `persons_only=True`, so its anomaly map covers only person
regions. This script runs consecutive-frame clips, thresholds the person anomaly
map, and draws red boxes on anomalous people.

Examples:
    # Calibration pass: scores only, no video
    .\.venv\Scripts\python.exe src\render_people_video.py --scene 01 --clip 01_0026 --score-only

    # Render
    .\.venv\Scripts\python.exe src\render_people_video.py --scene 01 --clip 01_0026 \
        --image-threshold 1.2 --box-threshold 1.05
"""

from __future__ import annotations

import argparse
import csv
import json
import time
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--scene", default="01")
    parser.add_argument("--clip", default="01_0026")
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--image-threshold", type=float, default=0.7)
    parser.add_argument("--box-threshold", type=float, default=0.55)
    parser.add_argument("--relative-threshold", type=float, default=0.75)
    parser.add_argument("--motion-threshold", type=float, default=0.45)
    parser.add_argument("--score-mode", choices=["weighted", "sum"], default="weighted")
    parser.add_argument("--velocity-weight", type=float, default=0.6)
    parser.add_argument("--pose-weight", type=float, default=0.25)
    parser.add_argument("--deep-weight", type=float, default=0.15)
    parser.add_argument("--cluster-radius", type=float, default=0.0)
    parser.add_argument("--focus-radius", type=float, default=0.0)
    parser.add_argument("--focus-ratio", type=float, default=0.8)
    parser.add_argument("--focus-jump", type=float, default=1.6)
    parser.add_argument("--focus-timeout", type=int, default=30)
    parser.add_argument("--min-area", type=int, default=1200)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument(
        "--frame-gap",
        type=int,
        default=1,
        help="Frames between the first and last AI-VAD input frames",
    )
    parser.add_argument("--track-iou", type=float, default=0.3)
    parser.add_argument("--track-window", type=int, default=10)
    parser.add_argument("--track-min-hits", type=int, default=6)
    parser.add_argument("--track-stop-hits", type=int, default=10)
    parser.add_argument("--track-stop-threshold", type=float, default=None)
    parser.add_argument("--cooldown-frames", type=int, default=30)
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--end-frame", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_model(checkpoint: Path, device: str) -> AiVad:
    model = AiVad.load_from_checkpoint(
        str(checkpoint), map_location=device, post_processor=False
    )
    model.model.eval().to(device)
    return model


def to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def component_weights(args: argparse.Namespace) -> dict[str, float]:
    if args.score_mode == "sum":
        return {"velocity": 1.0, "pose": 1.0, "deep": 1.0}
    return {
        "velocity": args.velocity_weight,
        "pose": args.pose_weight,
        "deep": args.deep_weight,
    }


def infer_pair(
    model: AiVad,
    first_bgr: np.ndarray,
    last_bgr: np.ndarray,
    device: str,
    weights: dict[str, float],
):
    clip = torch.stack([to_tensor(first_bgr), to_tensor(last_bgr)]).unsqueeze(0).to(device)
    try:
        with torch.no_grad():
            first_frame = clip[:, 0]
            last_frame = clip[:, -1]
            flow = model.model.flow_extractor(first_frame, last_frame)
            regions = model.model.region_extractor(first_frame, last_frame)
            features = model.model.feature_extractor(first_frame, flow, regions)
            feature_set = features[0]
            estimator = model.model.density_estimator
            component_scores = {}
            if estimator.use_velocity_features and feature_set[FeatureType.VELOCITY].numel():
                component_scores["velocity"] = estimator.velocity_estimator.predict(
                    feature_set[FeatureType.VELOCITY]
                )
            if estimator.use_deep_features and feature_set[FeatureType.DEEP].numel():
                component_scores["deep"] = estimator.appearance_estimator.predict(
                    feature_set[FeatureType.DEEP]
                )
            if estimator.use_pose_features and feature_set[FeatureType.POSE].numel():
                component_scores["pose"] = estimator.pose_estimator.predict(
                    feature_set[FeatureType.POSE]
                )
            if not component_scores:
                height, width = last_bgr.shape[:2]
                return 0.0, np.zeros((height, width), dtype=np.float32), None, None, {}

            region_scores = sum(
                weights.get(name, 0.0) * values
                for name, values in component_scores.items()
            )
            image_scores = sum(
                weights.get(name, 0.0) * values.max()
                for name, values in component_scores.items()
            )
            anomaly_map = torch.amax(
                regions[0]["masks"] * region_scores.view(-1, 1, 1, 1),
                dim=0,
            )
    except IndexError:
        height, width = last_bgr.shape[:2]
        return 1.0, np.zeros((height, width), dtype=np.float32), None, None, {}
    score = float(image_scores.detach().cpu())
    anomaly_map = np.squeeze(anomaly_map)
    if anomaly_map.ndim != 2:
        raise RuntimeError(f"Unexpected anomaly map shape: {anomaly_map.shape}")
    return (
        score,
        anomaly_map.detach().float().cpu().numpy(),
        regions[0],
        region_scores.detach().float().cpu(),
        {
            name: values.detach().float().cpu()
            for name, values in component_scores.items()
        },
    )


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


def iou(left: Detection, right: Detection) -> float:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x + left.w, right.x + right.w)
    y2 = min(left.y + left.h, right.y + right.h)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = left.w * left.h + right.w * right.h - intersection
    return intersection / union if union else 0.0


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
        score = float(weighted[index]) if weighted is not None else float(
            anomaly_map[y1:y2, x1:x2].max()
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
    """Pick the people that deserve a box and return the new focus point.

    A person is boxed only when their own score is high *and* their motion score
    clears the motion threshold. When a focus point is active, a new high-scoring
    person far from the current event is rejected; this stops a passing walker from
    stealing the box from a fighter.
    """
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
        top_distance = ((top_cx - focus_center[0]) ** 2 + (top_cy - focus_center[1]) ** 2) ** 0.5

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


def draw_overlay(frame, boxes, score, image_threshold):
    for detection in boxes:
        x, y, w, h = detection.box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    is_anomaly = score >= image_threshold and len(boxes) > 0
    label = "PEOPLE ANOMALY" if is_anomaly else "NORMAL"
    color = (0, 0, 255) if is_anomaly else (0, 180, 0)
    text = f"{label}  score={score:.2f}  people_boxes={len(boxes)}"
    cv2.rectangle(frame, (0, 0), (500, 38), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)
    return frame


def main() -> None:
    args = parse_args()
    frames_dir = args.frames_dir or (ROOT / "data/shanghaitech/testing/frames" / args.clip)
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if args.limit > 0:
        frame_paths = frame_paths[: args.limit]
    if len(frame_paths) < 2:
        raise SystemExit(f"Not enough frames in {frames_dir}")

    device = args.device if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    weights = component_weights(args)

    output = args.output or (
        ROOT / "results/videos" / f"shanghaitech_scene{args.scene}_{args.clip}_people.mp4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".scores.csv")
    gap = max(1, args.frame_gap)

    writer = None
    if not args.score_only:
        first = cv2.imread(str(frame_paths[0]))
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
        )
        if not writer.isOpened():
            raise SystemExit(f"Could not open writer for {output}")

    print(f"checkpoint : {args.checkpoint}")
    print(f"clip       : {args.clip}  ({len(frame_paths)} frames)")
    print(f"device     : {device}")
    print(f"frame gap  : {gap} frames ({gap / args.fps:.2f}s at {args.fps:g} fps)")
    if not args.score_only:
        print(f"output     : {output}")
        print(f"thresholds : image={args.image_threshold} box={args.box_threshold}")
    print(f"csv        : {csv_path}")

    total = len(frame_paths) - gap
    start = max(1, args.start_frame)
    end = args.end_frame if args.end_frame > 0 else total
    end = min(end, total)
    proc_start = max(1, start - args.smooth_window)

    rows = []
    history: list[float] = []
    tracks: dict[int, Track] = {}
    next_track_id = 1
    focus_center: tuple[float, float] | None = None
    last_focus_frame = -10_000
    prev = cv2.imread(str(frame_paths[proc_start - gap]))
    started = time.time()

    for idx in range(proc_start, end + 1):
        current = cv2.imread(str(frame_paths[idx]))
        first = cv2.imread(str(frame_paths[idx - gap]))
        raw_score, anomaly_map, region, region_scores, component_scores = infer_pair(
            model, first, current, device, weights
        )

        if not history:
            history = [raw_score] * max(1, args.smooth_window)
        history.append(raw_score)
        history = history[-max(1, args.smooth_window) :]
        score = sum(history) / len(history)

        candidates = score_persons(
            anomaly_map,
            region,
            region_scores,
            component_scores,
            args.min_area,
        )
        if idx - last_focus_frame > args.focus_timeout:
            focus_center = None
        detections, focus_center = select_persons(
            candidates,
            args.box_threshold,
            args.relative_threshold,
            args.motion_threshold,
            args.cluster_radius,
            args.top_k,
            focus_center=focus_center,
            focus_radius=args.focus_radius,
            focus_ratio=args.focus_ratio,
            focus_jump=args.focus_jump,
        )
        if score < args.image_threshold:
            detections = []
        elif detections:
            last_focus_frame = idx
        tracks, next_track_id, boxes, matched = update_tracks(
            tracks,
            detections,
            next_track_id,
            args.track_iou,
            max(1, args.track_window),
            max(1, args.track_min_hits),
            max(1, args.track_stop_hits),
            max(0, args.cooldown_frames),
            args.box_threshold,
            args.track_stop_threshold
            if args.track_stop_threshold is not None
            else args.box_threshold * 0.85,
        )

        rows.append(
            {
                "frame_index": idx,
                "frame_file": frame_paths[idx].name,
                "score": round(score, 4),
                "raw_score": round(raw_score, 4),
                "image_threshold": args.image_threshold,
                "box_threshold": args.box_threshold,
                "motion_threshold": args.motion_threshold,
                "label": "ANOMALY" if score >= args.image_threshold else "NORMAL",
                "boxes": len(boxes),
                "map_max": round(float(anomaly_map.max()), 4),
                "velocity_max": round(
                    float(component_scores["velocity"].max()), 4
                )
                if "velocity" in component_scores
                else 0.0,
                "pose_max": round(float(component_scores["pose"].max()), 4)
                if "pose" in component_scores
                else 0.0,
                "deep_max": round(float(component_scores["deep"].max()), 4)
                if "deep" in component_scores
                else 0.0,
                "active_track_ids": ",".join(
                    str(detection.track_id) for detection in boxes
                ),
                "matched_tracks": matched,
                "frame_gap": gap,
                "region_scores": json.dumps(
                    [round(float(value), 4) for value in region_scores.numpy()]
                    if region_scores is not None
                    else []
                ),
                "velocity_scores": json.dumps(
                    [round(float(value), 4) for value in component_scores["velocity"].numpy()]
                    if "velocity" in component_scores
                    else []
                ),
                "pose_scores": json.dumps(
                    [round(float(value), 4) for value in component_scores["pose"].numpy()]
                    if "pose" in component_scores
                    else []
                ),
                "deep_scores": json.dumps(
                    [round(float(value), 4) for value in component_scores["deep"].numpy()]
                    if "deep" in component_scores
                    else []
                ),
                "candidates": json.dumps(
                    [
                        {
                            "box": [c["x"], c["y"], c["w"], c["h"]],
                            "area": c["area"],
                            "score": round(c["score"], 4),
                            "velocity": round(c["velocity"], 4),
                            "pose": round(c["pose"], 4),
                            "deep": round(c["deep"], 4),
                        }
                        for c in candidates
                    ]
                ),
            }
        )

        if writer is not None and idx >= start:
            frame = draw_overlay(current, boxes, score, args.image_threshold)
            writer.write(frame)

        if idx % 25 == 0 or idx == end:
            elapsed = time.time() - started
            rate = (idx - proc_start + 1) / elapsed if elapsed else 0
            print(
                f"  {idx}/{end}  raw={raw_score:.2f}  smooth={score:.2f}  "
                f"boxes={len(boxes)}  map_max={anomaly_map.max():.3f}  {rate:.2f} fps",
                flush=True,
            )
        prev = current

    if writer is not None:
        writer.release()

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys())
        csv_writer = csv.DictWriter(handle, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    if not args.score_only:
        sample_dir = output.parent / f"{args.clip}_people_samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for old in sample_dir.glob("*.jpg"):
            old.unlink()
        chosen = set()
        chosen.add(min(rows, key=lambda r: r["score"])["frame_index"])
        chosen.add(max(rows, key=lambda r: r["score"])["frame_index"])
        flagged = [r for r in rows if r["boxes"] > 0]
        if flagged:
            chosen.add(flagged[0]["frame_index"])
            chosen.add(flagged[len(flagged) // 2]["frame_index"])
        score_by_frame = {int(row["frame_index"]): float(row["score"]) for row in rows}
        for idx in sorted(chosen):
            image = cv2.imread(str(frame_paths[idx]))
            prev_frame = cv2.imread(str(frame_paths[idx - gap]))
            _, anomaly_map, region, region_scores, component_scores = infer_pair(
                model, prev_frame, image, device, weights
            )
            score = score_by_frame.get(idx, 0.0)
            candidates = score_persons(
                anomaly_map,
                region,
                region_scores,
                component_scores,
                args.min_area,
            )
            boxes, _ = select_persons(
                candidates,
                args.box_threshold,
                args.relative_threshold,
                args.motion_threshold,
                args.cluster_radius,
                args.top_k,
            )
            if score < args.image_threshold:
                boxes = []
            cv2.imwrite(
                str(sample_dir / f"frame_{idx:03d}.jpg"),
                draw_overlay(image, boxes, score, args.image_threshold),
            )
        print(f"samples    : {sample_dir}")

    print(f"done       : {len(rows)} frames in {time.time() - started:.1f}s")
    if not args.score_only:
        print(f"video      : {output}")
    print(f"scores     : {csv_path}")


if __name__ == "__main__":
    main()
