from __future__ import annotations

import numpy as np
import pytest

from src.evaluate_fps import (
    aggregate_rows,
    event_ranges,
    f1,
    match_events,
    metrics_pair,
)


def make_rows(frames, scores, labels=None, boxes=None):
    labels = labels or ["NORMAL"] * len(frames)
    boxes = boxes or [0] * len(frames)
    return [
        {
            "frame_index": frame,
            "time_s": frame / 10.0,
            "score": score,
            "label": label,
            "boxes": box,
        }
        for frame, score, label, box in zip(frames, scores, labels, boxes, strict=True)
    ]


def test_event_ranges():
    assert event_ranges([1, 2, 3, 7, 9, 10]) == [(1, 3), (7, 7), (9, 10)]
    assert event_ranges([]) == []


def test_match_events_with_tolerance():
    reference = [(10, 20), (40, 50)]
    assert match_events(reference, [(11, 19), (55, 60)], tolerance_frames=2) == 1
    assert match_events(reference, [(10, 20), (40, 50)], tolerance_frames=0) == 2


def test_f1():
    assert f1(8, 2, 2) == pytest.approx(0.8)
    assert f1(0, 0, 0) == 0.0


def test_aggregate_rows_collapses_upsampled_frames():
    rows = make_rows(
        [1, 1, 1, 2],
        [0.2, 0.4, 0.6, 0.5],
        ["NORMAL", "NORMAL", "ANOMALY", "NORMAL"],
        [0, 0, 1, 0],
    )
    aggregated = aggregate_rows(rows)
    assert set(aggregated) == {1, 2}
    assert aggregated[1]["score"] == pytest.approx(0.4)
    assert aggregated[1]["boxes"] == 1
    assert aggregated[1]["label"] == "ANOMALY"


def test_identical_runs_are_perfectly_invariant():
    frames = list(range(1, 21))
    scores = [0.1 * (i % 7) + i / 100 for i in frames]
    labels = ["ANOMALY" if 8 <= frame <= 12 else "NORMAL" for frame in frames]
    boxes = [1 if label == "ANOMALY" else 0 for label in labels]
    run = aggregate_rows(make_rows(frames, scores, labels, boxes))

    metrics = metrics_pair(run, run, fps=10.0)
    assert metrics["frames"] == 20
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["pearson"] == pytest.approx(1.0)
    assert metrics["alarm_jaccard"] == 1.0
    assert metrics["box_jaccard"] == 1.0
    assert metrics["event_matches"] == metrics["ref_events"]
    assert metrics["first_alarm_delta_s"] == 0.0


def test_shifted_alarm_reports_delay_in_seconds():
    frames = list(range(1, 21))
    scores = [float(frame) for frame in frames]
    reference_labels = ["ANOMALY" if 8 <= frame <= 12 else "NORMAL" for frame in frames]
    shifted_labels = ["ANOMALY" if 10 <= frame <= 12 else "NORMAL" for frame in frames]
    reference = aggregate_rows(make_rows(frames, scores, reference_labels))
    shifted = aggregate_rows(make_rows(frames, scores, shifted_labels))

    metrics = metrics_pair(reference, shifted, fps=10.0)
    assert metrics["first_alarm_delta_s"] == 0.2
    assert metrics["last_alarm_delta_s"] == 0.0
    assert metrics["alarm_jaccard"] < 1.0


def test_gt_metrics_are_computed():
    frames = list(range(0, 20))
    gt = np.zeros(30, dtype=int)
    gt[10:15] = 1
    reference = aggregate_rows(
        make_rows(frames, [1.0 if 10 <= frame < 15 else 0.0 for frame in frames])
    )
    other = aggregate_rows(
        make_rows(frames, [0.0 if 10 <= frame < 15 else 1.0 for frame in frames])
    )
    metrics = metrics_pair(reference, other, fps=10.0, gt=gt)
    assert metrics["ref_auroc"] == 1.0
    assert metrics["run_auroc"] == 0.0
