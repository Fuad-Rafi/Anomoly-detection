"""Tune the per-person score composition and alarm threshold on a labelled clip.

The renderer logs every person candidate with its velocity, pose and deep scores.
This script recombines those components offline, so no GPU pass is needed for each
setting, and reports the frame-level precision/recall/F1 of every combination.

Examples:
    .\\.venv\\Scripts\\python.exe src\\tune_people_score.py `
        --csv results/videos/gap5_calibration.scores.csv `
        --gt data/shanghaitech/testing/test_frame_mask/01_0026.npy
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--min-area", type=int, default=1200)
    return parser.parse_args()


def f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main() -> None:
    args = parse_args()
    gt = np.load(args.gt).squeeze()
    rows = list(csv.DictReader(args.csv.open(encoding="utf-8")))
    frames = np.array([int(row["frame_index"]) for row in rows])
    labels = np.array([gt[min(frame, len(gt) - 1)] for frame in frames]).astype(int)
    candidates_per_frame = [json.loads(row["candidates"]) for row in rows]

    weight_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    thresholds = np.arange(0.05, 1.55, 0.05)
    motion_gates = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

    results = []
    for velocity_weight, pose_weight, deep_weight in itertools.product(
        weight_grid, weight_grid, weight_grid
    ):
        total = velocity_weight + pose_weight + deep_weight
        if total == 0:
            continue
        weights = np.array([velocity_weight, pose_weight, deep_weight]) / total

        frame_scores = np.zeros(len(frames))
        for index, candidates in enumerate(candidates_per_frame):
            if not candidates:
                continue
            values = np.array(
                [
                    [
                        candidate["velocity"],
                        candidate["pose"],
                        candidate["deep"],
                    ]
                    for candidate in candidates
                    if candidate["area"] >= args.min_area
                ]
            )
            if len(values) == 0:
                continue
            frame_scores[index] = float(np.max(values @ weights))

        for motion_gate in motion_gates:
            if motion_gate > 0:
                gated_scores = np.zeros(len(frames))
                for index, candidates in enumerate(candidates_per_frame):
                    values = np.array(
                        [
                            [
                                candidate["velocity"],
                                candidate["pose"],
                                candidate["deep"],
                            ]
                            for candidate in candidates
                            if candidate["area"] >= args.min_area
                            and candidate["velocity"] >= motion_gate
                        ]
                    )
                    if len(values):
                        gated_scores[index] = float(np.max(values @ weights))
                scores = gated_scores
            else:
                scores = frame_scores

            for threshold in thresholds:
                predicted = scores >= threshold
                tp = int(((predicted == 1) & (labels == 1)).sum())
                fp = int(((predicted == 1) & (labels == 0)).sum())
                fn = int(((predicted == 0) & (labels == 1)).sum())
                score = f1(tp, fp, fn)
                if score > 0:
                    results.append(
                        (
                            score,
                            float(threshold),
                            tuple(np.round(weights, 3)),
                            motion_gate,
                            tp,
                            fp,
                            fn,
                        )
                    )

    results.sort(key=lambda item: -item[0])
    print(f"rows {len(frames)}, GT positives {int(labels.sum())}")
    print("f1    thr   weights(v,p,d)        gate  tp  fp  fn")
    for score, threshold, weights, motion_gate, tp, fp, fn in results[: args.top]:
        print(
            f"{score:.3f} {threshold:4.2f}  {str(weights):<22s} {motion_gate:4.2f} "
            f"{tp:3d} {fp:3d} {fn:3d}"
        )


if __name__ == "__main__":
    main()
