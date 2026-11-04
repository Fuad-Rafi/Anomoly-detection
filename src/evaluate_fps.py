"""Compare AI-VAD score runs and report frame-rate invariance metrics.

Runs are aligned on the **source frame index** (not on output sample index), so a
30 fps emulation of a 10 fps source is compared against the frames it actually saw.
All temporal deltas are reported in seconds.

Example:
    .\\.venv\\Scripts\\python.exe src\\evaluate_fps.py ^
        --reference fps10 ^
        --run fps5=results/fps/shanghaitech/01_0026/fps5_dt0p2083/scores.csv ^
        --run fps20=results/fps/shanghaitech/01_0026/fps20_dt0p2083/scores.csv ^
        --run fps30=results/fps/shanghaitech/01_0026/fps30_dt0p2083/scores.csv ^
        --gt data/shanghaitech/testing/test_frame_mask/01_0026.npy ^
        --output results/fps/invariance_01_0026.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

try:
    from .temporal import frames_to_seconds
except ImportError:  # pragma: no cover - script import
    from temporal import frames_to_seconds  # type: ignore[no-redef]


def f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def event_ranges(frames: list[int]) -> list[tuple[int, int]]:
    """Contiguous runs of frame indices (inclusive)."""
    if not frames:
        return []
    frames = sorted(frames)
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


def match_events(
    reference: list[tuple[int, int]],
    other: list[tuple[int, int]],
    tolerance_frames: int,
) -> int:
    """Count events in ``other`` whose start is within tolerance of a reference event."""
    used: set[int] = set()
    matches = 0
    for start, _ in other:
        for index, (ref_start, _) in enumerate(reference):
            if index in used:
                continue
            if abs(start - ref_start) <= tolerance_frames:
                used.add(index)
                matches += 1
                break
    return matches


def aggregate_rows(rows: list[dict]) -> dict[int, dict]:
    """Collapse repeated source frames (upsampled runs) to one row per frame."""
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        frame = int(row["frame_index"])
        grouped.setdefault(frame, []).append(row)

    aggregated = {}
    for frame, items in grouped.items():
        scores = [float(item["score"]) for item in items]
        aggregated[frame] = {
            "frame_index": frame,
            "time_s": float(items[0]["time_s"]),
            "score": float(np.mean(scores)),
            "boxes": max(int(item["boxes"]) for item in items),
            "label": "ANOMALY"
            if any(item["label"] == "ANOMALY" for item in items)
            else "NORMAL",
        }
    return aggregated


def read_run(path: Path) -> dict[int, dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return aggregate_rows(list(csv.DictReader(handle)))


def _safe_correlation(function, left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return float("nan")
    value = function(left, right)[0]
    return float(value)


def metrics_pair(
    reference: dict[int, dict],
    other: dict[int, dict],
    *,
    fps: float | None = None,
    gt: np.ndarray | None = None,
    tolerance_frames: int = 5,
) -> dict:
    frames = sorted(set(reference) & set(other))
    if not frames:
        return {"frames": 0}

    ref_scores = np.asarray([reference[f]["score"] for f in frames], dtype=float)
    other_scores = np.asarray([other[f]["score"] for f in frames], dtype=float)
    frame_set = set(frames)

    ref_alarms = {
        frame for frame in frame_set if reference[frame]["label"] == "ANOMALY"
    }
    other_alarms = {
        frame for frame in frame_set if other[frame]["label"] == "ANOMALY"
    }
    ref_boxes = {frame for frame in frame_set if reference[frame]["boxes"] > 0}
    other_boxes = {frame for frame in frame_set if other[frame]["boxes"] > 0}

    alarm_union = ref_alarms | other_alarms
    alarm_jaccard = len(ref_alarms & other_alarms) / len(alarm_union) if alarm_union else 1.0
    box_union = ref_boxes | other_boxes
    box_jaccard = len(ref_boxes & other_boxes) / len(box_union) if box_union else 1.0

    ref_events = event_ranges(sorted(ref_alarms))
    other_events = event_ranges(sorted(other_alarms))

    def first_last(times: set[int]) -> tuple[float, float]:
        if not times:
            return (float("nan"), float("nan"))
        return (min(times), max(times))

    ref_first, ref_last = first_last(ref_alarms)
    other_first, other_last = first_last(other_alarms)
    fps_value = fps or 10.0

    result = {
        "frames": len(frames),
        "spearman": _safe_correlation(spearmanr, ref_scores, other_scores),
        "pearson": _safe_correlation(pearsonr, ref_scores, other_scores),
        "ks": float(ks_2samp(ref_scores, other_scores).statistic),
        "median_ratio": float(
            np.median(other_scores[ref_scores != 0] / ref_scores[ref_scores != 0])
        )
        if np.any(ref_scores != 0)
        else float("nan"),
        "score_mae": float(np.mean(np.abs(ref_scores - other_scores))),
        "alarm_jaccard": float(alarm_jaccard),
        "box_jaccard": float(box_jaccard),
        "ref_alarm_frames": len(ref_alarms),
        "run_alarm_frames": len(other_alarms),
        "ref_box_frames": len(ref_boxes),
        "run_box_frames": len(other_boxes),
        "ref_events": len(ref_events),
        "run_events": len(other_events),
        "event_matches": match_events(ref_events, other_events, tolerance_frames),
        "first_alarm_delta_s": frames_to_seconds(other_first - ref_first, fps_value)
        if not (np.isnan(ref_first) or np.isnan(other_first))
        else float("nan"),
        "last_alarm_delta_s": frames_to_seconds(other_last - ref_last, fps_value)
        if not (np.isnan(ref_last) or np.isnan(other_last))
        else float("nan"),
    }

    if gt is not None:
        label_by_frame = {
            frame: int(gt[min(frame, len(gt) - 1)]) for frame in frames
        }
        labels = np.asarray([label_by_frame[frame] for frame in frames], dtype=int)
        if labels.min() != labels.max():
            result["ref_auroc"] = float(roc_auc_score(labels, ref_scores))
            result["run_auroc"] = float(roc_auc_score(labels, other_scores))
        else:
            result["ref_auroc"] = float("nan")
            result["run_auroc"] = float("nan")

        for prefix, run in (("ref", reference), ("run", other)):
            alarms = [
                frame for frame in frames if run[frame]["label"] == "ANOMALY"
            ]
            tp = sum(1 for frame in alarms if label_by_frame[frame] == 1)
            fp = len(alarms) - tp
            fn = int(labels.sum()) - tp
            result[f"{prefix}_f1"] = f1(tp, fp, fn)
            duration_s = max(run[f]["time_s"] for f in run) - min(
                run[f]["time_s"] for f in run
            )
            result[f"{prefix}_fa_per_min"] = (
                float(fp) / (duration_s / 60.0) if duration_s > 0 else float("nan")
            )

    return result


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("runs must be NAME=CSV")
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--gt", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--tolerance-s", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = {name: read_run(path) for name, path in args.run}
    if args.reference not in runs:
        raise SystemExit(f"Reference {args.reference!r} is not one of {sorted(runs)}")
    reference = runs[args.reference]
    gt = np.load(args.gt).squeeze().astype(int) if args.gt else None
    fps = args.fps or 10.0
    tolerance_frames = max(1, int(round(args.tolerance_s * fps)))

    print(
        f"reference : {args.reference}  frames={len(reference)}"
        + (f"  gt={args.gt}" if args.gt else "")
    )
    print(
        "run       frames  spearman  pearson     ks  med_ratio  alarm_J  box_J  "
        "alarms  boxes  events  matched  first_delta_s  auroc  f1  fa/min"
    )
    rows = []
    for name, run in runs.items():
        if name == args.reference:
            continue
        metrics = metrics_pair(
            reference,
            run,
            fps=fps,
            gt=gt,
            tolerance_frames=tolerance_frames,
        )
        row = {"reference": args.reference, "run": name, **metrics}
        rows.append(row)
        print(
            f"{name:<10s} {metrics.get('frames', 0):6d}  "
            f"{metrics.get('spearman', float('nan')):8.3f}  "
            f"{metrics.get('pearson', float('nan')):7.3f}  "
            f"{metrics.get('ks', float('nan')):5.3f}  "
            f"{metrics.get('median_ratio', float('nan')):9.3f}  "
            f"{metrics.get('alarm_jaccard', float('nan')):7.3f}  "
            f"{metrics.get('box_jaccard', float('nan')):5.3f}  "
            f"{metrics.get('run_alarm_frames', 0):6d}  "
            f"{metrics.get('run_box_frames', 0):5d}  "
            f"{metrics.get('run_events', 0):6d}  "
            f"{metrics.get('event_matches', 0):7d}  "
            f"{metrics.get('first_alarm_delta_s', float('nan')):13.2f}  "
            f"{metrics.get('run_auroc', float('nan')):5.3f}  "
            f"{metrics.get('run_f1', float('nan')):4.2f}  "
            f"{metrics.get('run_fa_per_min', float('nan')):6.2f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nsummary   : {args.output}")


if __name__ == "__main__":
    main()
