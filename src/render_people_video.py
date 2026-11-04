"""Render a people-only anomaly video with AI-VAD.

AI-VAD is fitted with ``persons_only=True``, so its anomaly map covers only person
regions.  This script runs frame pairs, thresholds the person anomaly map, and draws
red boxes on anomalous people.

The temporal window is defined in seconds (``--target-dt``).  The legacy
``--frame-gap`` flag remains available and is used when ``--target-dt`` is omitted.

Examples:
    # Calibration pass: scores only, no video
    .\\.venv\\Scripts\\python.exe src\\render_people_video.py --scene 01 --clip 01_0026 --score-only

    # Render
    .\\.venv\\Scripts\\python.exe src\\render_people_video.py --scene 01 --clip 01_0026 \
        --image-threshold 1.2 --box-threshold 1.05

    # Time-based invocation (Δt = 0.208 s at 10 fps -> gap 2)
    .\\.venv\\Scripts\\python.exe src\\render_people_video.py --scene 01 --clip 01_0026 \
        --source-fps 10 --target-dt 0.208 --smooth-seconds 1.5
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

try:  # package import (pytest / -m src...)
    from .aivad_core import (
        DEFAULT_CKPT,
        component_weights,
        infer_pair,
        load_model,
        score_persons,
        select_persons,
        update_tracks,
    )
    from .temporal import effective_dt, gap_for, seconds_to_frames
except ImportError:  # script import (python src/render_people_video.py)
    from aivad_core import (  # type: ignore[no-redef]
        DEFAULT_CKPT,
        component_weights,
        infer_pair,
        load_model,
        score_persons,
        select_persons,
        update_tracks,
    )
    from temporal import effective_dt, gap_for, seconds_to_frames  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--scene", default="01")
    parser.add_argument("--clip", default="01_0026")
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=10.0, help="Output video fps")
    parser.add_argument(
        "--source-fps",
        type=float,
        default=10.0,
        help="Native fps of the frame source (used for seconds conversion)",
    )
    parser.add_argument(
        "--target-dt",
        type=float,
        default=None,
        help="Temporal window in seconds; overrides --frame-gap",
    )
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
    parser.add_argument(
        "--focus-timeout-seconds",
        type=float,
        default=None,
        help="Override --focus-timeout with a duration",
    )
    parser.add_argument("--min-area", type=int, default=1200)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument(
        "--smooth-seconds",
        type=float,
        default=None,
        help="Override --smooth-window with a duration",
    )
    parser.add_argument(
        "--frame-gap",
        type=int,
        default=1,
        help="Frames between the first and last AI-VAD input frames (legacy)",
    )
    parser.add_argument("--track-iou", type=float, default=0.3)
    parser.add_argument("--track-window", type=int, default=10)
    parser.add_argument(
        "--track-window-seconds",
        type=float,
        default=None,
        help="Override --track-window with a duration",
    )
    parser.add_argument("--track-min-hits", type=int, default=6)
    parser.add_argument(
        "--track-min-hits-seconds",
        type=float,
        default=None,
        help="Override --track-min-hits with a duration",
    )
    parser.add_argument("--track-stop-hits", type=int, default=10)
    parser.add_argument(
        "--track-stop-hits-seconds",
        type=float,
        default=None,
        help="Override --track-stop-hits with a duration",
    )
    parser.add_argument("--track-stop-threshold", type=float, default=None)
    parser.add_argument("--cooldown-frames", type=int, default=30)
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=None,
        help="Override --cooldown-frames with a duration",
    )
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--end-frame", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _seconds_or_frames(args: argparse.Namespace) -> dict[str, int]:
    """Resolve the temporal parameters, preferring the seconds-based flags."""
    fps = args.source_fps

    def resolve(seconds: float | None, frames: int) -> int:
        if seconds is None:
            return max(1, int(frames))
        return max(1, seconds_to_frames(seconds, fps))

    return {
        "smooth": resolve(args.smooth_seconds, args.smooth_window),
        "focus_timeout": resolve(args.focus_timeout_seconds, args.focus_timeout),
        "cooldown": resolve(args.cooldown_seconds, args.cooldown_frames),
        "track_window": resolve(args.track_window_seconds, args.track_window),
        "min_hits": resolve(args.track_min_hits_seconds, args.track_min_hits),
        "stop_hits": resolve(args.track_stop_hits_seconds, args.track_stop_hits),
    }


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

    if args.target_dt is not None:
        gap = gap_for(args.source_fps, args.target_dt)
    else:
        gap = max(1, args.frame_gap)

    timing = _seconds_or_frames(args)
    smooth_window = timing["smooth"]
    focus_timeout = timing["focus_timeout"]
    cooldown_frames = timing["cooldown"]
    track_window = timing["track_window"]
    min_hits = timing["min_hits"]
    stop_hits = timing["stop_hits"]

    device = args.device if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    weights = component_weights(
        args.score_mode, args.velocity_weight, args.pose_weight, args.deep_weight
    )

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
    print(
        f"frame gap  : {gap} frames "
        f"({effective_dt(args.source_fps, gap):.3f}s at {args.source_fps:g} fps)"
    )
    print(
        f"timing     : smooth={smooth_window} focus_timeout={focus_timeout} "
        f"cooldown={cooldown_frames} track_window={track_window} "
        f"min_hits={min_hits} stop_hits={stop_hits}"
    )
    if not args.score_only:
        print(f"output     : {output}")
        print(f"thresholds : image={args.image_threshold} box={args.box_threshold}")
    print(f"csv        : {csv_path}")

    total = len(frame_paths) - gap
    start = max(gap, args.start_frame)
    end = args.end_frame if args.end_frame > 0 else total
    end = min(end, total)
    if end < start:
        raise SystemExit(
            f"Not enough frames for gap {gap}: start={start}, end={end}, "
            f"frames={len(frame_paths)}"
        )
    # Never let the first frame index go negative when gap > 1 (the legacy
    # renderer used gap 1, where max(1, ...) was sufficient).
    proc_start = max(gap, start - smooth_window)

    rows = []
    history: list[float] = []
    tracks: dict = {}
    next_track_id = 1
    focus_center: tuple[float, float] | None = None
    last_focus_frame = -10_000
    started = time.time()

    for idx in range(proc_start, end + 1):
        current = cv2.imread(str(frame_paths[idx]))
        first = cv2.imread(str(frame_paths[idx - gap]))
        raw_score, anomaly_map, region, region_scores, component_scores = infer_pair(
            model, first, current, device, weights
        )

        if not history:
            history = [raw_score] * max(1, smooth_window)
        history.append(raw_score)
        history = history[-max(1, smooth_window) :]
        score = sum(history) / len(history)

        candidates = score_persons(
            anomaly_map,
            region,
            region_scores,
            component_scores,
            args.min_area,
        )
        if idx - last_focus_frame > focus_timeout:
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
            max(1, track_window),
            max(1, min_hits),
            max(1, stop_hits),
            max(0, cooldown_frames),
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
