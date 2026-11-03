"""Offline test of persistent spatial-cluster tracking for people-only alarms.

Per-person scores cannot separate a fighter from a walker when the fighter is
being pushed and the walker is striding. The interaction is the reliable cue:
the fighter pair stays in one small cluster for many frames, while walkers pass
through. This script clusters people per frame, links clusters across frames,
and selects the cluster with the strongest persistent evidence.

Usage:
    .\\.venv\\Scripts\\python.exe src\\analyze_cluster_tracks.py `
        --csv results/videos/gap5train_gap2_calibration.scores.csv `
        --gt data/shanghaitech/testing/test_frame_mask/01_0026.npy
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Cluster:
    members: list[dict]
    centroid: tuple[float, float]
    score: float
    pose: float
    velocity: float


@dataclass
class ClusterTrack:
    track_id: int
    cluster: Cluster
    age: int = 1
    missed: int = 0
    score_ema: float = 0.0
    history: list[float] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--min-area", type=int, default=1200)
    parser.add_argument("--cluster-distance", type=float, default=100.0)
    parser.add_argument("--match-distance", type=float, default=100.0)
    parser.add_argument("--max-missed", type=int, default=5)
    parser.add_argument("--min-age", type=int, default=10)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--min-pose", type=float, default=0.0)
    parser.add_argument("--age-weight", type=float, default=0.01)
    parser.add_argument("--ema", type=float, default=0.3)
    parser.add_argument("--top", type=int, default=12)
    return parser.parse_args()


def center(candidate: dict) -> tuple[float, float]:
    x, y, w, h = candidate["box"]
    return x + w / 2, y + h / 2


def distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2) ** 0.5


def make_clusters(candidates: list[dict], threshold: float) -> list[Cluster]:
    if not candidates:
        return []
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
        if distance(center(candidates[left]), center(candidates[right])) <= threshold:
            union(left, right)

    grouped: dict[int, list[dict]] = {}
    for index, candidate in enumerate(candidates):
        grouped.setdefault(find(index), []).append(candidate)

    clusters = []
    for members in grouped.values():
        xs = [center(member)[0] for member in members]
        ys = [center(member)[1] for member in members]
        clusters.append(
            Cluster(
                members=members,
                centroid=(float(np.mean(xs)), float(np.mean(ys))),
                score=float(np.mean([member["score"] for member in members])),
                pose=float(np.mean([member["pose"] for member in members])),
                velocity=float(np.mean([member["velocity"] for member in members])),
            )
        )
    return clusters


def evaluate(rows, gt, args) -> tuple[list[int], int, float, float, float]:
    tracks: dict[int, ClusterTrack] = {}
    next_id = 1
    boxed = []
    for row in rows:
        frame = int(row["frame_index"])
        candidates = [
            candidate
            for candidate in json.loads(row["candidates"])
            if candidate["area"] >= args.min_area
        ]
        frame_clusters = make_clusters(candidates, args.cluster_distance)

        unmatched_tracks = set(tracks)
        unmatched_clusters = set(range(len(frame_clusters)))
        matches = []
        for track_id, track in tracks.items():
            best = max(
                (
                    (distance(track.cluster.centroid, cluster.centroid), index)
                    for index, cluster in enumerate(frame_clusters)
                    if index in unmatched_clusters
                ),
                default=(1e9, -1),
            )
            if best[0] <= args.match_distance:
                matches.append((track_id, best[1]))
                unmatched_tracks.discard(track_id)
                unmatched_clusters.discard(best[1])

        for track_id, cluster_index in matches:
            track = tracks[track_id]
            cluster = frame_clusters[cluster_index]
            track.cluster = cluster
            track.age += 1
            track.missed = 0
            track.score_ema = (
                args.ema * cluster.score + (1 - args.ema) * track.score_ema
            )
            track.history = (track.history + [cluster.score])[-20:]

        for track_id in unmatched_tracks:
            tracks[track_id].missed += 1
            if tracks[track_id].missed > args.max_missed:
                del tracks[track_id]

        for cluster_index in unmatched_clusters:
            cluster = frame_clusters[cluster_index]
            tracks[next_id] = ClusterTrack(
                next_id, cluster, score_ema=cluster.score, history=[cluster.score]
            )
            next_id += 1

        options = []
        for track in tracks.values():
            if track.missed:
                continue
            if track.age < args.min_age:
                continue
            if track.cluster.score < args.min_score:
                continue
            if track.cluster.pose < args.min_pose:
                continue
            selection = track.score_ema + args.age_weight * min(track.age, 50)
            options.append((selection, track))
        if options:
            options.sort(key=lambda item: item[0], reverse=True)
            boxed.append(frame)

    labels = np.array([gt[min(int(row["frame_index"]), len(gt) - 1)] for row in rows])
    tp = sum(1 for frame in boxed if gt[frame] == 1)
    precision = tp / len(boxed) if boxed else 0.0
    recall = tp / max(1, int(labels.sum()))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return boxed, tp, precision, recall, f1


def main() -> None:
    args = parse_args()
    gt = np.load(args.gt).squeeze()
    rows = list(csv.DictReader(args.csv.open(encoding="utf-8")))
    boxed, tp, precision, recall, f1 = evaluate(rows, gt, args)
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
    print(f"boxed frames : {len(boxed)}  on GT {tp}")
    print(f"precision {precision:.3f}  recall {recall:.3f}  f1 {f1:.3f}")
    print(f"ranges       : {groups}")


if __name__ == "__main__":
    main()
