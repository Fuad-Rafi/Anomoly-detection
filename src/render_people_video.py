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
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from anomalib.models import AiVad

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
    parser.add_argument("--image-threshold", type=float, default=1.3)
    parser.add_argument("--box-threshold", type=float, default=1.1)
    parser.add_argument("--relative-threshold", type=float, default=0.95)
    parser.add_argument("--cluster-radius", type=float, default=150.0)
    parser.add_argument("--min-area", type=int, default=1200)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--smooth-window", type=int, default=15)
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


def infer_pair(model: AiVad, first_bgr: np.ndarray, last_bgr: np.ndarray, device: str):
    clip = torch.stack([to_tensor(first_bgr), to_tensor(last_bgr)]).unsqueeze(0).to(device)
    try:
        with torch.no_grad():
            first_frame = clip[:, 0]
            last_frame = clip[:, -1]
            flow = model.model.flow_extractor(first_frame, last_frame)
            regions = model.model.region_extractor(first_frame, last_frame)
            features = model.model.feature_extractor(first_frame, flow, regions)
            region_scores, image_scores = model.model.density_estimator(features[0])
            anomaly_map = torch.amax(
                regions[0]["masks"] * region_scores.view(-1, 1, 1, 1),
                dim=0,
            )
    except IndexError:
        height, width = last_bgr.shape[:2]
        return 1.0, np.zeros((height, width), dtype=np.float32), None, None
    score = float(image_scores.detach().cpu())
    anomaly_map = np.squeeze(anomaly_map)
    if anomaly_map.ndim != 2:
        raise RuntimeError(f"Unexpected anomaly map shape: {anomaly_map.shape}")
    return (
        score,
        anomaly_map.detach().float().cpu().numpy(),
        regions[0],
        region_scores.detach().float().cpu(),
    )


def person_boxes(
    anomaly_map: np.ndarray,
    region: dict | None,
    region_scores: torch.Tensor | None,
    threshold: float,
    relative_threshold: float,
    cluster_radius: float,
    min_area: int,
    top_k: int,
) -> list[tuple[int, int, int, int]]:
    if region is None or "boxes" not in region or len(region["boxes"]) == 0:
        return []

    height, width = anomaly_map.shape
    scored = []
    boxes = region["boxes"].cpu().numpy()
    scores = (
        region_scores.numpy()
        if region_scores is not None and len(region_scores) == len(boxes)
        else None
    )
    for box_index, box in enumerate(boxes):
        x1, y1, x2, y2 = (int(value) for value in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        area = (x2 - x1) * (y2 - y1)
        if area < min_area:
            continue
        score = float(scores[box_index]) if scores is not None else float(
            anomaly_map[y1:y2, x1:x2].max()
        )
        scored.append((score, area, x1, y1, x2 - x1, y2 - y1))

    if not scored:
        return []

    top_score, _, top_x, top_y, top_w, top_h = max(scored, key=lambda item: item[0])
    top_cx, top_cy = top_x + top_w / 2, top_y + top_h / 2
    cutoff = max(threshold, relative_threshold * top_score)

    selected = []
    for item in scored:
        score, _, x, y, w, h = item
        cx, cy = x + w / 2, y + h / 2
        near = ((cx - top_cx) ** 2 + (cy - top_cy) ** 2) ** 0.5 <= cluster_radius
        if score >= cutoff or (near and score >= threshold):
            selected.append(item)

    selected.sort(key=lambda item: (-item[0], -item[1]))
    return [(x, y, w, h) for _, _, x, y, w, h in selected[:top_k]]


def draw_overlay(frame, boxes, score, image_threshold):
    for x, y, w, h in boxes:
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

    output = args.output or (
        ROOT / "results/videos" / f"shanghaitech_scene{args.scene}_{args.clip}_people.mp4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".scores.csv")

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
    if not args.score_only:
        print(f"output     : {output}")
        print(f"thresholds : image={args.image_threshold} box={args.box_threshold}")
    print(f"csv        : {csv_path}")

    total = len(frame_paths) - 1
    start = max(1, args.start_frame)
    end = args.end_frame if args.end_frame > 0 else total
    end = min(end, total)
    proc_start = max(1, start - args.smooth_window)

    rows = []
    history: list[float] = []
    prev = cv2.imread(str(frame_paths[proc_start - 1]))
    started = time.time()

    for idx in range(proc_start, end + 1):
        current = cv2.imread(str(frame_paths[idx]))
        raw_score, anomaly_map, region, region_scores = infer_pair(
            model, prev, current, device
        )

        if not history:
            history = [raw_score] * max(1, args.smooth_window)
        history.append(raw_score)
        history = history[-max(1, args.smooth_window) :]
        score = sum(history) / len(history)

        boxes = person_boxes(
            anomaly_map,
            region,
            region_scores,
            args.box_threshold,
            args.relative_threshold,
            args.cluster_radius,
            args.min_area,
            args.top_k,
        )
        if score < args.image_threshold:
            boxes = []

        rows.append(
            {
                "frame_index": idx,
                "frame_file": frame_paths[idx].name,
                "score": round(score, 4),
                "raw_score": round(raw_score, 4),
                "image_threshold": args.image_threshold,
                "box_threshold": args.box_threshold,
                "label": "ANOMALY" if score >= args.image_threshold else "NORMAL",
                "boxes": len(boxes),
                "map_max": round(float(anomaly_map.max()), 4),
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
            prev_frame = cv2.imread(str(frame_paths[idx - 1]))
            _, anomaly_map, region, region_scores = infer_pair(
                model, prev_frame, image, device
            )
            score = score_by_frame.get(idx, 0.0)
            boxes = person_boxes(
                anomaly_map,
                region,
                region_scores,
                args.box_threshold,
                args.relative_threshold,
                args.cluster_radius,
                args.min_area,
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
