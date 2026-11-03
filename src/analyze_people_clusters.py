"""Offline test of a spatial-cluster rule for people-only alert boxes.

A fighter pair produces several overlapping person boxes in one small area; a
passer-by produces an isolated box. This script clusters candidate boxes by
center distance, scores each cluster by pose and weighted anomaly score, and
selects the strongest interacting group. It reports frame precision/recall
against the ground-truth mask.

Usage:
    .\\.venv\\Scripts\\python.exe src\\analyze_people_clusters.py `
        --csv results/videos/gap5train_gap2_calibration.scores.csv `
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
    parser.add_argument("--min-area", type=int, default=1200)
    parser.add_argument("--cluster-distance", type=float, default=100.0)
    parser.add_argument("--min-pose", type=float, default=0.0)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--min-size", type=int, default=2)
    parser.add_argument("--pose-weight", type=float, default=1.0)
    parser.add_argument("--score-weight", type=float, default=0.5)
    parser.add_argument("--top", type=int, default=12)
    return parser.parse_args()


def center(candidate: dict) -> tuple[float, float]:
    x, y, w, h = candidate["box"]
    return x + w / 2, y + h / 2


def distance(left: dict, right: dict) -> float:
    lx, ly = center(left)
    rx, ry = center(right)
    return ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5


def clusters(candidates: list[dict], threshold: float) -> list[list[dict]]:
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in itertools.combinations(range(len(candidates)), 2):
        if distance(candidates[left], candidates[right]) <= threshold:
            union(left, right)

    grouped: dict[int, list[dict]] = {}
    for index, candidate in enumerate(candidates):
        grouped.setdefault(find(index), []).append(candidate)
    return list(grouped.values())


def main() -> None:
    args = parse_args()
    gt = np.load(args.gt).squeeze()
    rows = list(csv.DictReader(args.csv.open(encoding="utf-8")))

    boxed = []
    for row in rows:
        frame = int(row["frame_index"])
        candidates = [
            candidate
            for candidate in json.loads(row["candidates"])
            if candidate["area"] >= args.min_area
        ]
        options = []
        for group in clusters(candidates, args.cluster_distance):
            if len(group) < args.min_size:
                continue
            mean_pose = float(np.mean([c["pose"] for c in group]))
            mean_score = float(np.mean([c["score"] for c in group]))
            cluster_score = args.pose_weight * mean_pose + args.score_weight * mean_score
            if mean_pose < args.min_pose or mean_score < args.min_score:
                continue
            options.append((cluster_score, mean_pose, mean_score, len(group), group))
        if options:
            options.sort(key=lambda item: item[0], reverse=True)
            boxed.append(frame)

    labels = np.array([gt[min(int(row["frame_index"]), len(gt) - 1)] for row in rows])
    tp = sum(1 for frame in boxed if gt[frame] == 1)
    precision = tp / len(boxed) if boxed else 0.0
    recall = tp / max(1, int(labels.sum()))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    groups = []
    if boxed:
        start = previous = boxed[0]
        for frame in boxed[1:]:
            if frame == previous + 1:
                previous = frame
            else:
                groups.append((start, previous))
                start = previous = frame
        groups.append((start, previous))
    print(f"boxed frames : {len(boxed)}")
    print(f"boxed on GT  : {tp}  precision {precision:.3f}  recall {recall:.3f}  f1 {f1:.3f}")
    print(f"ranges       : {groups}")


if __name__ == "__main__":
    main()
