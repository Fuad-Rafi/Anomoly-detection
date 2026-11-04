"""Calibrate AI-VAD thresholds on held-out normal footage.

Example:
    .\.venv\Scripts\python.exe src\calibrate_people_thresholds.py \
        --frames-dir data\shanghaitech\testing\frames\01_0001 \
        --frame-gap 5 --output results\calibration\normal_01_0001.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from .aivad_core import DEFAULT_CKPT, infer_pair, load_model
    from .temporal import gap_for
except ImportError:  # script import
    from aivad_core import DEFAULT_CKPT, infer_pair, load_model  # type: ignore[no-redef]
    from temporal import gap_for  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--video", type=Path, action="append", default=[])
    parser.add_argument("--video-glob")
    parser.add_argument(
        "--interval",
        action="append",
        default=[],
        help="Optional video interval as path:start:end (inclusive end)",
    )
    parser.add_argument("--max-pairs-per-video", type=int, default=200)
    parser.add_argument("--sample-step", type=int, default=1)
    parser.add_argument("--frame-gap", type=int, default=1)
    parser.add_argument("--source-fps", type=float, default=10.0)
    parser.add_argument("--target-dt", type=float, default=None)
    parser.add_argument("--quantile", type=float, default=99.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_dt is not None:
        gap = gap_for(args.source_fps, args.target_dt)
    else:
        gap = max(1, args.frame_gap)
    if not args.frames_dir and not args.video and not args.video_glob and not args.interval:
        raise SystemExit("Provide --frames-dir, --video, or --video-glob")

    device = args.device if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    image_scores: list[float] = []
    region_scores: list[float] = []
    component_scores: dict[str, list[float]] = {
        "velocity": [],
        "pose": [],
        "deep": [],
    }

    sources = []
    if args.frames_dir:
        sources.append(("frames", sorted(args.frames_dir.glob("*.jpg"))))
    sources.extend(("video", [path]) for path in args.video)
    if args.video_glob:
        sources.extend(("video", [Path(path)]) for path in sorted(glob.glob(args.video_glob)))
    for specification in args.interval:
        path_text, start_text, end_text = specification.rsplit(":", 2)
        path = Path(path_text)
        if path.is_dir():
            sources.append(
                ("frames_interval", [sorted(path.glob("*.jpg")), int(start_text), int(end_text)])
            )
        else:
            sources.append(("video_interval", [path, int(start_text), int(end_text)]))

    for source_type, source in sources:
        if source_type in {"frames", "frames_interval"}:
            cap = None
            length = len(source)
            if source_type == "frames_interval":
                length = len(source[0])
        else:
            cap = cv2.VideoCapture(str(source[0]))
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if length <= gap:
            if cap:
                cap.release()
            continue
        interval_start = gap
        interval_end = length - 1
        if source_type in {"video_interval", "frames_interval"}:
            interval_start = max(gap, source[1])
            interval_end = min(length - 1, source[2])
        indices = range(interval_start, interval_end + 1, max(1, args.sample_step))
        if args.max_pairs_per_video > 0:
            indices = list(indices)[: args.max_pairs_per_video]
        for index in indices:
            if source_type in {"frames", "frames_interval"}:
                frame_source = source if source_type == "frames" else source[0]
                first = cv2.imread(str(frame_source[index - gap]))
                current = cv2.imread(str(frame_source[index]))
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index - gap)
                ok_first, first = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok_current, current = cap.read()
                if not ok_first or not ok_current:
                    continue
            image_score, _, _, regions, components = infer_pair(
                model, first, current, device
            )
            image_scores.append(image_score)
            if regions is not None and len(regions):
                region_scores.extend(float(value) for value in regions.numpy())
            for name, values in components.items():
                component_scores[name].extend(float(value) for value in values.numpy())
        if cap:
            cap.release()

    def percentile(values: list[float]) -> float:
        return float(np.percentile(values, args.quantile)) if values else 0.0

    source_names = [str(args.frames_dir)] if args.frames_dir else [
        str(path) for path in args.video
    ]
    source_names.extend(args.interval)
    result = {
        "sources": source_names,
        "frame_gap": gap,
        "pairs": len(image_scores),
        "quantile": args.quantile,
        "image_threshold": percentile(image_scores),
        "box_threshold": percentile(region_scores),
        "component_thresholds": {
            name: percentile(values) for name, values in component_scores.items()
        },
        "image_max": max(image_scores, default=0.0),
        "region_max": max(region_scores, default=0.0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
