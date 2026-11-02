"""Calibrate AI-VAD thresholds on held-out normal footage.

Example:
    .\.venv\Scripts\python.exe src\calibrate_people_thresholds.py \
        --frames-dir data\shanghaitech\testing\frames\01_0001 \
        --frame-gap 5 --output results\calibration\normal_01_0001.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from render_people_video import DEFAULT_CKPT, infer_pair, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--frame-gap", type=int, default=1)
    parser.add_argument("--quantile", type=float, default=99.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_paths = sorted(args.frames_dir.glob("*.jpg"))
    gap = max(1, args.frame_gap)
    if len(frame_paths) <= gap:
        raise SystemExit(f"Need more than {gap} frames in {args.frames_dir}")

    device = args.device if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    image_scores: list[float] = []
    region_scores: list[float] = []
    component_scores: dict[str, list[float]] = {
        "velocity": [],
        "pose": [],
        "deep": [],
    }

    for index in range(gap, len(frame_paths)):
        first = cv2.imread(str(frame_paths[index - gap]))
        current = cv2.imread(str(frame_paths[index]))
        image_score, _, _, regions, components = infer_pair(
            model, first, current, device
        )
        image_scores.append(image_score)
        if regions is not None and len(regions):
            region_scores.extend(float(value) for value in regions.numpy())
        for name, values in components.items():
            component_scores[name].extend(float(value) for value in values.numpy())

    def percentile(values: list[float]) -> float:
        return float(np.percentile(values, args.quantile)) if values else 0.0

    result = {
        "frames_dir": str(args.frames_dir),
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
