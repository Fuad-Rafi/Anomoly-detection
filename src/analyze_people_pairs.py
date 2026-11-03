"""Offline test of an interaction-pair rule for people-only alert boxes.

The renderer currently boxes people by individual score, which lets a passer-by
with a high pose score steal a box from a fighter. Fights are persistent close
interactions, so this script tracks candidate people and selects the pair that
stays closest for the longest time, then reports frame precision/recall against
the frame mask.

Usage:
    .\\.venv\\Scripts\\python.exe src\\analyze_people_pairs.py `
        --csv results/videos/gap5train_gap2_calibration.scores.csv `
        --gt data/shanghaitech/testing/test_frame_mask/01_0026.npy
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Candidate:
    x: int
    y: int
    w: int
    h: int
    score: float
    velocity: float
    pose: float
    deep: float

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2


@dataclass
class Track:
    track_id: int
    candidate: Candidate
    age: int = 1
    missed: int = 0
    history: list[Candidate] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--min-area", type=int, default=1200)
    parser.add_argument("--gate", type=float, default=90.0)
    parser.add_argument("--pair-distance", type=float, default=110.0)
    parser.add_argument("--min-persistence", type=int, default=8)
    parser.add_argument("--max-missed", type=int, default=8)
    parser.add_argument("--min-score", type=float, default=0.16)
    parser.add_argument("--min-pose", type=float, default=0.0)
    return parser.parse_args()


def distance(a: Candidate, b: Candidate) -> float:
    ax, ay = a.center
    bx, by = b.center
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def main() -> None:
    args = parse_args()
    gt = np.load(args.gt).squeeze()
    rows = list(csv.DictReader(args.csv.open(encoding="utf-8")))

    tracks: dict[int, Track] = {}
    next_id = 1
    pair_persistence: dict[tuple[int, int], int] = {}
    boxed_frames = []
    selected_by_frame = {}

    for row in rows:
        frame = int(row["frame_index"])
        raw = json.loads(row["candidates"])
        candidates = [
            Candidate(
                x=c["box"][0],
                y=c["box"][1],
                w=c["box"][2],
                h=c["box"][3],
                score=c["score"],
                velocity=c["velocity"],
                pose=c["pose"],
                deep=c["deep"],
            )
            for c in raw
            if c["area"] >= args.min_area
        ]

        unmatched_tracks = set(tracks)
        unmatched_candidates = set(range(len(candidates)))
        for track_id, track in tracks.items():
            best = max(
                (
                    (distance(track.candidate, candidate), index)
                    for index, candidate in enumerate(candidates)
                    if index in unmatched_candidates
                ),
                default=(1e9, -1),
            )
            if best[0] <= args.gate:
                candidate = candidates[best[1]]
                track.candidate = candidate
                track.age += 1
                track.missed = 0
                track.history = (track.history + [candidate])[-20:]
                unmatched_tracks.discard(track_id)
                unmatched_candidates.discard(best[1])

        for track_id in unmatched_tracks:
            tracks[track_id].missed += 1
            if tracks[track_id].missed > args.max_missed:
                del tracks[track_id]

        for index in unmatched_candidates:
            tracks[next_id] = Track(next_id, candidates[index], history=[candidates[index]])
            next_id += 1

        active = [track for track in tracks.values() if track.missed == 0]
        visible_pairs = []
        for i, left in enumerate(active):
            for right in active[i + 1 :]:
                if distance(left.candidate, right.candidate) > args.pair_distance:
                    continue
                key = tuple(sorted((left.track_id, right.track_id)))
                pair_persistence[key] = pair_persistence.get(key, 0) + 1
                pair_score = (left.candidate.score + right.candidate.score) / 2
                pair_pose = (left.candidate.pose + right.candidate.pose) / 2
                if (
                    pair_persistence[key] >= args.min_persistence
                    and pair_score >= args.min_score
                    and pair_pose >= args.min_pose
                ):
                    visible_pairs.append(
                        (pair_persistence[key], pair_score, left, right)
                    )

        if visible_pairs:
            visible_pairs.sort(key=lambda item: (item[0], item[1]), reverse=True)
            _, _, left, right = visible_pairs[0]
            selected_by_frame[frame] = [left.candidate, right.candidate]
            boxed_frames.append(frame)

    labels = np.array([gt[min(int(row["frame_index"]), len(gt) - 1)] for row in rows])
    tp = sum(1 for frame in boxed_frames if gt[frame] == 1)
    precision = tp / len(boxed_frames) if boxed_frames else 0.0
    recall = tp / max(1, int(labels.sum()))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    groups = []
    if boxed_frames:
        start = previous = boxed_frames[0]
        for frame in boxed_frames[1:]:
            if frame == previous + 1:
                previous = frame
            else:
                groups.append((start, previous))
                start = previous = frame
        groups.append((start, previous))

    print(f"boxed frames : {len(boxed_frames)}")
    print(f"boxed on GT  : {tp}  precision {precision:.3f}  recall {recall:.3f}  f1 {f1:.3f}")
    print(f"ranges       : {groups}")
    print(f"gt span      : {int(np.argmax(gt))}-{len(gt) - 1 - int(np.argmax(gt[::-1]))}")


if __name__ == "__main__":
    main()
