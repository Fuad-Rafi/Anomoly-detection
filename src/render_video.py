"""Render an annotated anomaly-detection video.

Loads a fitted Anomalib PatchCore checkpoint, runs frame-by-frame inference on a
ShanghaiTech clip, and draws red boxes around regions whose anomaly map exceeds the
checkpoint's pixel threshold.

Example:
    .\.venv\Scripts\python.exe src\render_video.py --scene 01 --clip 01_0136
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from anomalib.models import Patchcore

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = (
    ROOT
    / "results/patchcore/scene01/Patchcore/ShanghaiTech/v0/weights/lightning/model.ckpt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--scene", default="01", help="Scene prefix, e.g. 01")
    parser.add_argument("--clip", default="01_0136", help="Clip name, e.g. 01_0136")
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Override frame directory (defaults to data/shanghaitech/testing/frames/<clip>)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output MP4 path (defaults to results/videos/<clip>_annotated.mp4)",
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Pixel threshold for boxes (default: checkpoint pixel_threshold)",
    )
    parser.add_argument(
        "--image-threshold",
        type=float,
        default=None,
        help="Frame threshold for the ANOMALY/NORMAL banner (default: checkpoint)",
    )
    parser.add_argument("--min-area", type=int, default=120, help="Minimum box area in pixels")
    parser.add_argument("--top-k", type=int, default=3, help="Maximum boxes per frame")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="Only render N frames (0 = all)")
    parser.add_argument("--no-samples", action="store_true", help="Skip sample frame export")
    return parser.parse_args()


def load_model(checkpoint: Path, device: str) -> tuple[Patchcore, torch.Tensor, torch.Tensor]:
    model = Patchcore.load_from_checkpoint(str(checkpoint), map_location=device)
    model.to(device).eval()
    model.model.eval()

    pixel_threshold = model.post_processor.pixel_threshold.detach().float()
    image_threshold = model.post_processor.image_threshold.detach().float()
    return model, pixel_threshold, image_threshold


def infer_frame(model: Patchcore, transform, frame_bgr: np.ndarray, device: str):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    tensor = transform(tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model.model(tensor)

    score = float(output.pred_score[0].detach().cpu())
    anomaly_map = output.anomaly_map[0, 0].detach().float().cpu().numpy()
    return score, anomaly_map


def anomaly_boxes(
    anomaly_map: np.ndarray,
    threshold: float,
    out_w: int,
    out_h: int,
    min_area: int,
    top_k: int,
) -> list[tuple[int, int, int, int]]:
    mask = (anomaly_map > threshold).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area >= min_area:
            candidates.append((area, x, y, w, h))
    candidates.sort(reverse=True)
    return [(x, y, w, h) for _, x, y, w, h in candidates[:top_k]]


def draw_overlay(
    frame: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    score: float,
    image_threshold: float,
) -> np.ndarray:
    for x, y, w, h in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    is_anomaly = score >= image_threshold
    label = "ANOMALY" if is_anomaly else "NORMAL"
    color = (0, 0, 255) if is_anomaly else (0, 180, 0)
    text = f"{label}  score={score:.1f}  boxes={len(boxes)}"

    cv2.rectangle(frame, (0, 0), (460, 38), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame


def main() -> None:
    args = parse_args()
    frames_dir = args.frames_dir or (
        ROOT / "data/shanghaitech/testing/frames" / args.clip
    )
    output = args.output or (ROOT / "results/videos" / f"{args.clip}_annotated.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not frames_dir.is_dir():
        raise SystemExit(f"Frames directory not found: {frames_dir}")

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise SystemExit(f"No .jpg frames in {frames_dir}")
    if args.limit > 0:
        frame_paths = frame_paths[: args.limit]

    device = args.device if torch.cuda.is_available() else "cpu"
    model, pixel_threshold_t, image_threshold_t = load_model(args.checkpoint, device)
    pixel_threshold = float(pixel_threshold_t) if args.threshold is None else args.threshold
    image_threshold = (
        float(image_threshold_t) if args.image_threshold is None else args.image_threshold
    )
    transform = model.pre_processor.transform

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise SystemExit(f"Could not read {frame_paths[0]}")
    height, width = first.shape[:2]

    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
    )
    if not writer.isOpened():
        raise SystemExit(f"Could not open video writer for {output}")

    print(f"checkpoint : {args.checkpoint}")
    print(f"frames     : {len(frame_paths)} from {frames_dir}")
    print(f"output     : {output}  ({width}x{height} @ {args.fps} fps)")
    print(f"thresholds : pixel={pixel_threshold:.3f} image={image_threshold:.3f}")
    print(f"device     : {device}")

    rows = []
    samples: list[tuple[int, int, float]] = []  # (frame_idx, box_count, score)
    started = time.time()

    for idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"warning: unreadable frame {frame_path}", flush=True)
            continue

        score, anomaly_map = infer_frame(model, transform, frame, device)
        boxes = anomaly_boxes(
            anomaly_map,
            pixel_threshold,
            frame.shape[1],
            frame.shape[0],
            args.min_area,
            args.top_k,
        )
        if score < image_threshold:
            boxes = []
        draw_overlay(frame, boxes, score, image_threshold)
        writer.write(frame)

        rows.append(
            {
                "frame_index": idx,
                "frame_file": frame_path.name,
                "score": round(score, 4),
                "image_threshold": round(image_threshold, 4),
                "pixel_threshold": round(pixel_threshold, 4),
                "label": "ANOMALY" if score >= image_threshold else "NORMAL",
                "boxes": len(boxes),
                "max_box_area": max((w * h for _, _, w, h in boxes), default=0),
            }
        )
        samples.append((idx, len(boxes), score))

        if idx % 25 == 0 or idx == len(frame_paths) - 1:
            elapsed = time.time() - started
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  {idx + 1}/{len(frame_paths)}  "
                f"score={score:.1f}  boxes={len(boxes)}  {rate:.1f} fps",
                flush=True,
            )

    writer.release()

    csv_path = output.with_suffix(".scores.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys())
        csv_writer = csv.DictWriter(handle, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    if not args.no_samples:
        sample_dir = output.parent / f"{args.clip}_samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for old_sample in sample_dir.glob("*.jpg"):
            old_sample.unlink()
        chosen = set()
        if samples:
            chosen.add(min(samples, key=lambda s: s[2])[0])  # lowest score
            chosen.add(max(samples, key=lambda s: s[2])[0])  # highest score
            flagged = [s for s in samples if s[1] > 0]
            if flagged:
                chosen.add(flagged[0][0])
                chosen.add(flagged[len(flagged) // 2][0])
        for idx in sorted(chosen):
            src = frame_paths[idx]
            image = cv2.imread(str(src))
            score, anomaly_map = infer_frame(model, transform, image, device)
            boxes = anomaly_boxes(
                anomaly_map,
                pixel_threshold,
                image.shape[1],
                image.shape[0],
                args.min_area,
                args.top_k,
            )
            if score < image_threshold:
                boxes = []
            draw_overlay(image, boxes, score, image_threshold)
            cv2.imwrite(str(sample_dir / f"frame_{idx:03d}.jpg"), image)
        print(f"samples    : {sample_dir}")

    elapsed = time.time() - started
    print(f"done       : {len(rows)} frames in {elapsed:.1f}s")
    print(f"video      : {output}")
    print(f"scores     : {csv_path}")


if __name__ == "__main__":
    main()
