"""Headless, timestamped AI-VAD evaluation.

This is the scorer used by the frame-rate experiments.  It reads frames at a chosen
source rate, optionally emulates another frame rate, and writes one row per detector
invocation with both the requested and the *effective* temporal window.

Inference is cached per unique ``(first_frame, current_frame)`` pair.  Upsampled
20/30 fps runs therefore reuse the 10 fps pairs while still advancing the temporal
state machine at the emulated cadence.

Example:
    .\\.venv\\Scripts\\python.exe src\\eval_aivad.py `
        --dataset shanghaitech --clip 01_0026 `
        --checkpoint results/aivad/scene01_gap5/AiVad/ShanghaiTech/v0/weights/lightning/model_robust.ckpt `
        --source-fps 10 --emulate-fps 10 --target-dt 0.2083 `
        --image-threshold 1.4 --box-threshold 1.0 --score-mode sum `
        --output results/fps/shanghaitech/01_0026/fps10_dt0p2083/scores.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

try:  # package import
    from .aivad_core import (
        ROBUST_GAP5_CKPT,
        component_weights,
        infer_batch,
        load_model,
        score_persons,
        select_persons,
        update_tracks,
    )
    from .temporal import (
        FolderFrameSource,
        VideoFrameSource,
        plan_samples,
        seconds_to_frames,
        summarize_plan,
    )
except ImportError:  # script import
    from aivad_core import (  # type: ignore[no-redef]
        ROBUST_GAP5_CKPT,
        component_weights,
        infer_batch,
        load_model,
        score_persons,
        select_persons,
        update_tracks,
    )
    from temporal import (  # type: ignore[no-redef]
        FolderFrameSource,
        VideoFrameSource,
        plan_samples,
        seconds_to_frames,
        summarize_plan,
    )

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PairRecord:
    """Everything the temporal state machine needs from one unique frame pair."""

    score: float
    map_max: float
    component_max: dict[str, float] = field(default_factory=dict)
    region_scores_json: str = "[]"
    velocity_json: str = "[]"
    pose_json: str = "[]"
    deep_json: str = "[]"
    candidates: list[dict] = field(default_factory=list)


def candidates_json(candidates: list[dict]) -> str:
    """Serialise candidate boxes for the CSV, keeping the legacy 'box' shape."""
    return json.dumps(
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
    )


def draw_detection_overlay(
    frame: np.ndarray,
    boxes,
    score: float,
    image_threshold: float,
    effective_dt: float,
    *,
    banner: bool = True,
) -> np.ndarray:
    """Draw detection boxes and an optional score banner on a BGR frame."""
    for detection in boxes:
        cv2.rectangle(
            frame,
            (detection.x, detection.y),
            (detection.x + detection.w, detection.y + detection.h),
            (0, 0, 255),
            2,
        )
    if not banner:
        return frame
    is_anomaly = score >= image_threshold and len(boxes) > 0
    label = "ANOMALY" if is_anomaly else "NORMAL"
    color = (0, 0, 255) if is_anomaly else (0, 180, 0)
    text = (
        f"AI-VAD {label}  score={score:.2f}  boxes={len(boxes)}  "
        f"dt={effective_dt:.2f}s"
    )
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2
    )
    cv2.rectangle(
        frame,
        (0, 0),
        (min(frame.shape[1], text_width + 20), text_height + 18),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        frame,
        text,
        (10, text_height + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["shanghaitech", "avenue"], default="shanghaitech")
    parser.add_argument("--scene", default="01")
    parser.add_argument("--clip", default="01_0026")
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=ROBUST_GAP5_CKPT)
    parser.add_argument("--source-fps", type=float, default=None)
    parser.add_argument(
        "--emulate-fps",
        type=float,
        default=None,
        help="Output cadence for frame-rate experiments (default: source fps)",
    )
    parser.add_argument(
        "--target-dt",
        type=float,
        default=0.2083,
        help="Physical window between the two AI-VAD frames, in seconds",
    )
    parser.add_argument(
        "--process-hz",
        type=float,
        default=0.0,
        help="Decision cadence; 0 means process every emulated frame",
    )
    parser.add_argument("--start-s", type=float, default=None)
    parser.add_argument("--end-s", type=float, default=None)
    parser.add_argument(
        "--min-source-index",
        type=int,
        default=None,
        help="First source frame to score (default: 1 for folders, 0 for videos)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of samples")
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Process repeated (first,current) pairs at the emulated cadence",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)

    # Scoring / selection (defaults match the final people-only demo at 10 fps).
    parser.add_argument("--score-mode", choices=["weighted", "sum"], default="sum")
    parser.add_argument("--velocity-weight", type=float, default=0.6)
    parser.add_argument("--pose-weight", type=float, default=0.25)
    parser.add_argument("--deep-weight", type=float, default=0.15)
    parser.add_argument("--image-threshold", type=float, default=1.4)
    parser.add_argument("--box-threshold", type=float, default=1.0)
    parser.add_argument("--relative-threshold", type=float, default=0.5)
    parser.add_argument("--motion-threshold", type=float, default=0.45)
    parser.add_argument("--cluster-radius", type=float, default=90.0)
    parser.add_argument("--focus-radius", type=float, default=140.0)
    parser.add_argument("--focus-ratio", type=float, default=0.8)
    parser.add_argument("--focus-jump", type=float, default=1.6)
    parser.add_argument("--min-area", type=int, default=1200)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--track-iou", type=float, default=0.3)

    # Temporal parameters are durations, so they survive frame-rate changes.
    parser.add_argument("--smooth-seconds", type=float, default=1.5)
    parser.add_argument("--focus-timeout-seconds", type=float, default=3.0)
    parser.add_argument("--cooldown-seconds", type=float, default=3.0)
    parser.add_argument("--track-window-seconds", type=float, default=1.0)
    parser.add_argument("--track-min-hits-seconds", type=float, default=0.6)
    parser.add_argument("--track-stop-hits-seconds", type=float, default=1.0)
    parser.add_argument("--track-stop-threshold", type=float, default=None)

    # Optional annotated video output (used by the Avenue 25 fps demo).
    parser.add_argument("--render-output", type=Path, default=None)
    parser.add_argument(
        "--render-fps",
        type=float,
        default=None,
        help="Output video fps (default: source fps)",
    )
    parser.add_argument(
        "--no-render-banner",
        action="store_true",
        help="Do not draw the score banner",
    )
    return parser.parse_args()


def open_source(args: argparse.Namespace):
    if args.video is not None:
        source = VideoFrameSource(args.video, fps=args.source_fps)
        return source, "video", 0
    if args.frames_dir is not None:
        fps = args.source_fps
        if fps is None:
            fps = 10.0 if args.dataset == "shanghaitech" else None
        if fps is None:
            raise SystemExit("--source-fps is required for folder sources")
        source = FolderFrameSource(args.frames_dir, fps=fps)
        return source, "folder", 1
    if args.dataset == "shanghaitech":
        fps = args.source_fps or 10.0
        source = FolderFrameSource(
            ROOT / "data/shanghaitech/testing/frames" / args.clip, fps=fps
        )
        return source, "folder", 1
    if args.dataset == "avenue":
        name = args.clip if args.clip.endswith(".avi") else f"{args.clip}.avi"
        source = VideoFrameSource(
            ROOT / "data/avenue/testing_videos" / name, fps=args.source_fps
        )
        return source, "video", 0
    raise SystemExit(f"Unsupported dataset {args.dataset}")


def infer_batch_with_retry(
    model,
    first_frames: list[np.ndarray],
    last_frames: list[np.ndarray],
    device: str,
    weights: dict[str, float],
    batch_size: int,
):
    """Batch inference with an OOM fallback that halves the batch."""
    try:
        return infer_batch(model, first_frames, last_frames, device, weights)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or len(first_frames) <= 1:
            raise
        torch.cuda.empty_cache()
        midpoint = len(first_frames) // 2
        return infer_batch_with_retry(
            model,
            first_frames[:midpoint],
            last_frames[:midpoint],
            device,
            weights,
            max(1, batch_size // 2),
        ) + infer_batch_with_retry(
            model,
            first_frames[midpoint:],
            last_frames[midpoint:],
            device,
            weights,
            max(1, batch_size // 2),
        )


def build_pair_cache(
    model,
    source,
    unique_pairs: list[tuple[int, int]],
    *,
    device: str,
    weights: dict[str, float],
    batch_size: int,
    min_area: int,
) -> dict[tuple[int, int], PairRecord]:
    """Run inference once per unique pair and keep compact per-pair evidence."""
    cache: dict[tuple[int, int], PairRecord] = {}
    started = time.time()
    for start in range(0, len(unique_pairs), batch_size):
        batch = unique_pairs[start : start + batch_size]
        first_frames = [source.get(first) for first, _ in batch]
        last_frames = [source.get(current) for _, current in batch]
        results = infer_batch_with_retry(
            model, first_frames, last_frames, device, weights, batch_size
        )
        for key, result in zip(batch, results, strict=True):
            candidates = score_persons(
                result.anomaly_map,
                result.region,
                result.region_scores,
                result.component_scores,
                min_area,
            )
            cache[key] = PairRecord(
                score=result.score,
                map_max=float(result.anomaly_map.max()),
                component_max={
                    name: float(values.max())
                    for name, values in result.component_scores.items()
                },
                region_scores_json=json.dumps(
                    [round(float(value), 4) for value in result.region_scores.numpy()]
                    if result.region_scores is not None
                    else []
                ),
                velocity_json=json.dumps(
                    [
                        round(float(value), 4)
                        for value in result.component_scores["velocity"].numpy()
                    ]
                    if "velocity" in result.component_scores
                    else []
                ),
                pose_json=json.dumps(
                    [
                        round(float(value), 4)
                        for value in result.component_scores["pose"].numpy()
                    ]
                    if "pose" in result.component_scores
                    else []
                ),
                deep_json=json.dumps(
                    [
                        round(float(value), 4)
                        for value in result.component_scores["deep"].numpy()
                    ]
                    if "deep" in result.component_scores
                    else []
                ),
                # Keep the internal x/y/w/h form for select_persons; serialise it
                # only when writing the CSV.
                candidates=candidates,
            )
        processed = min(start + len(batch), len(unique_pairs))
        if processed % 50 == 0 or processed == len(unique_pairs):
            elapsed = time.time() - started
            print(
                f"  inference {processed}/{len(unique_pairs)} "
                f"({processed / elapsed:.2f} pairs/s)",
                flush=True,
            )
    return cache


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    source, source_kind, default_min_index = open_source(args)
    emulate_fps = args.emulate_fps or source.fps
    min_source_index = (
        default_min_index if args.min_source_index is None else args.min_source_index
    )
    samples = plan_samples(
        len(source),
        source.fps,
        args.target_dt,
        emulate_fps=emulate_fps,
        process_hz=args.process_hz or None,
        start_s=args.start_s,
        end_s=args.end_s,
        min_source_index=min_source_index,
        dedupe=not args.no_dedupe,
    )
    if args.limit > 0:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit("No samples planned; check --source-fps / --target-dt / ranges")

    # Decision cadence is the emulated rate (or the explicit process rate), not the
    # source-time spacing: upsampled runs legitimately repeat source timestamps.
    cadence = args.process_hz if args.process_hz > 0 else emulate_fps
    smooth_frames = max(1, seconds_to_frames(args.smooth_seconds, cadence))
    focus_timeout_frames = max(1, seconds_to_frames(args.focus_timeout_seconds, cadence))
    cooldown_frames = max(1, seconds_to_frames(args.cooldown_seconds, cadence))
    track_window = max(1, seconds_to_frames(args.track_window_seconds, cadence))
    min_hits = max(1, seconds_to_frames(args.track_min_hits_seconds, cadence))
    stop_hits = max(1, seconds_to_frames(args.track_stop_hits_seconds, cadence))

    plan_summary = summarize_plan(samples, args.target_dt)
    output = args.output
    if output is None:
        fps_label = f"{emulate_fps:g}"
        dt_label = f"{args.target_dt:g}".replace(".", "p")
        output = (
            ROOT
            / "results/fps"
            / args.dataset
            / args.clip
            / f"fps{fps_label}_dt{dt_label}"
            / "scores.csv"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    weights = component_weights(
        args.score_mode, args.velocity_weight, args.pose_weight, args.deep_weight
    )
    model = load_model(args.checkpoint, device)

    unique_pairs = list(
        dict.fromkeys((sample.first_index, sample.current_index) for sample in samples)
    )

    print(f"source     : {source_kind}  fps={source.fps:g}  frames={len(source)}")
    print(
        f"plan       : samples={plan_summary['samples']} "
        f"unique_pairs={len(unique_pairs)} dt={args.target_dt:g}s effective="
        f"{plan_summary['effective_dt_mean']:.4f}s "
        f"[{plan_summary['effective_dt_min']:.4f}, {plan_summary['effective_dt_max']:.4f}] "
        f"repeated_source_frames={plan_summary['repeated_source_frames']} "
        f"self_pairs={plan_summary['self_pairs']}"
    )
    print(
        f"cadence    : {cadence:.2f} Hz  smooth={smooth_frames} "
        f"focus_timeout={focus_timeout_frames} cooldown={cooldown_frames} "
        f"track_window={track_window} min_hits={min_hits} stop_hits={stop_hits}"
    )
    print(f"device     : {device}  batch={args.batch_size}")
    print(f"output     : {output}")

    cache = build_pair_cache(
        model,
        source,
        unique_pairs,
        device=device,
        weights=weights,
        batch_size=max(1, args.batch_size),
        min_area=args.min_area,
    )

    writer = None
    render_fps = args.render_fps or source.fps
    if args.render_output is not None:
        args.render_output.parent.mkdir(parents=True, exist_ok=True)
        first_render_frame = source.get(samples[0].current_index)
        render_height, render_width = first_render_frame.shape[:2]
        writer = cv2.VideoWriter(
            str(args.render_output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            render_fps,
            (render_width, render_height),
        )
        if not writer.isOpened():
            raise SystemExit(f"Could not open video writer for {args.render_output}")
        print(
            f"video      : {args.render_output} "
            f"({render_width}x{render_height} @ {render_fps:g} fps)"
        )

    # Temporal state machine over every planned sample (fast; no GPU work here).
    history: list[float] = []
    tracks: dict = {}
    next_track_id = 1
    focus_center: tuple[float, float] | None = None
    last_focus_sample = -10_000_000
    rows: list[dict] = []
    started = time.time()

    for sample in samples:
        record = cache[(sample.first_index, sample.current_index)]
        raw_score = record.score
        if not history:
            history = [raw_score] * smooth_frames
        history.append(raw_score)
        history = history[-smooth_frames:]
        score = sum(history) / len(history)

        if sample.index - last_focus_sample > focus_timeout_frames:
            focus_center = None
        detections, focus_center = select_persons(
            record.candidates,
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
            last_focus_sample = sample.index

        tracks, next_track_id, boxes, matched = update_tracks(
            tracks,
            detections,
            next_track_id,
            args.track_iou,
            track_window,
            min_hits,
            stop_hits,
            cooldown_frames,
            args.box_threshold,
            args.track_stop_threshold
            if args.track_stop_threshold is not None
            else args.box_threshold * 0.85,
        )

        if writer is not None:
            frame = source.get(sample.current_index).copy()
            draw_detection_overlay(
                frame,
                boxes,
                score,
                args.image_threshold,
                sample.effective_dt,
                banner=not args.no_render_banner,
            )
            writer.write(frame)

        if source_kind == "folder":
            frame_file = source.paths[sample.current_index].name
        else:
            frame_file = f"{sample.current_index:06d}"

        component_max = record.component_max
        rows.append(
            {
                "sample_index": sample.index,
                "frame_index": sample.current_index,
                "first_index": sample.first_index,
                "frame_file": frame_file,
                "time_s": round(sample.time_s, 5),
                "effective_dt": round(sample.effective_dt, 5),
                "source_gap": sample.source_gap,
                "duplicate": int(sample.duplicate),
                "score": round(score, 4),
                "raw_score": round(raw_score, 4),
                "image_threshold": args.image_threshold,
                "box_threshold": args.box_threshold,
                "motion_threshold": args.motion_threshold,
                "label": "ANOMALY" if score >= args.image_threshold else "NORMAL",
                "boxes": len(boxes),
                "map_max": round(record.map_max, 4),
                "velocity_max": round(component_max.get("velocity", 0.0), 4),
                "pose_max": round(component_max.get("pose", 0.0), 4),
                "deep_max": round(component_max.get("deep", 0.0), 4),
                "active_track_ids": ",".join(
                    str(detection.track_id) for detection in boxes
                ),
                "matched_tracks": matched,
                "region_scores": record.region_scores_json,
                "velocity_scores": record.velocity_json,
                "pose_scores": record.pose_json,
                "deep_scores": record.deep_json,
                "candidates": candidates_json(record.candidates),
            }
        )

    if writer is not None:
        writer.release()
        print(f"video done : {args.render_output}")

    with output.open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    metadata = {
        "dataset": args.dataset,
        "clip": args.clip,
        "source": str(args.video or args.frames_dir or ""),
        "source_kind": source_kind,
        "source_fps": source.fps,
        "emulate_fps": emulate_fps,
        "process_hz": args.process_hz,
        "target_dt": args.target_dt,
        "plan": plan_summary,
        "unique_pairs": len(unique_pairs),
        "cadence_hz": cadence,
        "smooth_frames": smooth_frames,
        "focus_timeout_frames": focus_timeout_frames,
        "cooldown_frames": cooldown_frames,
        "track_window": track_window,
        "min_hits": min_hits,
        "stop_hits": stop_hits,
        "checkpoint": str(args.checkpoint),
        "device": device,
        "batch_size": args.batch_size,
        "score_mode": args.score_mode,
        "weights": weights,
        "image_threshold": args.image_threshold,
        "box_threshold": args.box_threshold,
        "torch": torch.__version__,
        "rows": len(rows),
        "render_output": str(args.render_output) if args.render_output else None,
        "render_fps": render_fps if writer is not None else None,
    }
    metadata_path = output.with_name("metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"rows       : {len(rows)}  ({time.time() - started:.1f}s temporal pass)")
    print(f"scores     : {output}")
    print(f"metadata   : {metadata_path}")


if __name__ == "__main__":
    main()
