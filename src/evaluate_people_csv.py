"""Evaluate a people-only render/score CSV against a frame-level ground-truth mask.

The renderer writes one row per processed frame with:
  - the smoothed image score and the frame label,
  - the number of alert boxes actually drawn,
  - every person candidate with its weighted, velocity, pose and deep scores.

This script reports frame-level precision/recall and boxed-frame precision, plus
the thresholds that would have maximized frame-level F1.

Examples:
    .\\.venv\\Scripts\\python.exe src\\evaluate_people_csv.py `
        --csv results/videos/gap1_calibration.scores.csv `
        --gt data/shanghaitech/testing/test_frame_mask/01_0026.npy
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--score-key", default="score")
    parser.add_argument("--candidate-key", default="score")
    return parser.parse_args()


def ranges(frames: list[int]) -> list[tuple[int, int]]:
    if not frames:
        return []
    groups = []
    start = previous = frames[0]
    for frame in frames[1:]:
        if frame == previous + 1:
            previous = frame
        else:
            groups.append((start, previous))
            start = previous = frame
    groups.append((start, previous))
    return groups


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

    print(f"csv        : {args.csv}")
    print(f"rows       : {len(rows)}  gt positives: {int(labels.sum())}")
    print(f"gt span    : {int(np.argmax(gt))}-{len(gt) - 1 - int(np.argmax(gt[::-1]))}")

    boxed = [int(row["frame_index"]) for row in rows if int(row["boxes"]) > 0]
    boxed_on_gt = sum(1 for frame in boxed if gt[frame] == 1)
    print(f"boxed frames: {len(boxed)}  on GT: {boxed_on_gt}  "
          f"precision: {boxed_on_gt / len(boxed) if boxed else 0:.3f}")
    print(f"boxed ranges: {ranges(boxed)}")

    if "label" in rows[0]:
        alarms = [int(row["frame_index"]) for row in rows if row["label"] == "ANOMALY"]
        tp = sum(1 for frame in alarms if gt[frame] == 1)
        fp = len(alarms) - tp
        fn = int(labels.sum()) - tp
        print(f"label alarms: {len(alarms)}  on GT: {tp}  "
              f"precision: {tp / len(alarms) if alarms else 0:.3f}  "
              f"recall: {tp / max(1, int(labels.sum())):.3f}  f1: {f1(tp, fp, fn):.3f}")

    # Candidate-level sweep: treat the best person in each frame as the frame score.
    if rows and rows[0].get("candidates"):
        best = []
        for row in rows:
            candidates = json.loads(row["candidates"])
            best.append(max(candidates, key=lambda c: c[args.candidate_key])[args.candidate_key]
                        if candidates else 0.0)
        best = np.array(best)
        print("\ncandidate threshold sweep (best person per frame):")
        print("  thr   tp   fp   fn   precision  recall  f1")
        for threshold in np.arange(0.05, 0.85, 0.05):
            predicted = best >= threshold
            tp = int(((predicted == 1) & (labels == 1)).sum())
            fp = int(((predicted == 1) & (labels == 0)).sum())
            fn = int(((predicted == 0) & (labels == 1)).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            print(f"  {threshold:4.2f} {tp:4d} {fp:4d} {fn:4d}   {precision:7.3f}  {recall:6.3f}  {f1(tp, fp, fn):.3f}")

    if rows and rows[0].get("candidates"):
        all_candidates = [
            candidate
            for row in rows
            for candidate in json.loads(row["candidates"])
        ]
        gt_candidates = [
            candidate
            for row in rows
            for candidate in json.loads(row["candidates"])
            if gt[int(row["frame_index"])] == 1
        ]
        normal_candidates = [
            candidate
            for row in rows
            for candidate in json.loads(row["candidates"])
            if gt[int(row["frame_index"])] == 0
        ]
        print("\nper-candidate distributions (median / p90 / max):")
        for key in ("score", "velocity", "pose", "deep", "area"):
            for name, subset in (("GT1", gt_candidates), ("GT0", normal_candidates)):
                values = np.array([candidate[key] for candidate in subset])
                if len(values):
                    print(f"  {key:9s} {name}: {np.median(values):7.3f} / "
                          f"{np.percentile(values, 90):7.3f} / {values.max():7.3f}  (n={len(values)})")


if __name__ == "__main__":
    main()
