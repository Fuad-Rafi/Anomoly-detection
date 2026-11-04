"""Derive AI-VAD thresholds from a normal-footage score CSV.

The CSV is produced by ``eval_aivad.py`` on footage that contains no anomalies.
Percentiles of the frame score and of each per-person component become the display
thresholds for all frame-rate runs, so the same operating point is used everywhere.

Example:
    .\\.venv\\Scripts\\python.exe src\\calibrate_from_csv.py ^
        --csv results/fps/calibration/normal_01_001.csv ^
        --quantile 99.5 ^
        --output results/fps/calibration/normal_01_001.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def quantile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--quantile", type=float, default=99.5)
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=None,
        help="Override the velocity percentile (e.g. 0 to disable the motion gate)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No rows in {args.csv}")

    frame_scores = [float(row["score"]) for row in rows]
    candidates = [
        candidate
        for row in rows
        for candidate in json.loads(row.get("candidates") or "[]")
    ]

    result = {
        "source_csv": str(args.csv),
        "quantile": args.quantile,
        "pairs": len(rows),
        "frame_candidates": len(candidates),
        "image_threshold": quantile(frame_scores, args.quantile),
        "box_threshold": quantile(
            [float(candidate["score"]) for candidate in candidates], args.quantile
        ),
        "motion_threshold": (
            args.motion_threshold
            if args.motion_threshold is not None
            else quantile(
                [float(candidate["velocity"]) for candidate in candidates],
                args.quantile,
            )
        ),
        "pose_threshold": quantile(
            [float(candidate["pose"]) for candidate in candidates], args.quantile
        ),
        "deep_threshold": quantile(
            [float(candidate["deep"]) for candidate in candidates], args.quantile
        ),
        "frame_score_p50": quantile(frame_scores, 50),
        "frame_score_max": max(frame_scores),
        "candidate_score_p50": quantile(
            [float(candidate["score"]) for candidate in candidates], 50
        ),
        "velocity_p50": quantile(
            [float(candidate["velocity"]) for candidate in candidates], 50
        ),
        "velocity_max": max(
            (float(candidate["velocity"]) for candidate in candidates), default=0.0
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
