from __future__ import annotations

import pytest

from src.temporal import (
    effective_dt,
    frames_to_seconds,
    gap_for,
    plan_cadence_hz,
    plan_samples,
    round_half_up,
    seconds_to_frames,
    summarize_plan,
)


@pytest.mark.parametrize(
    ("fps", "dt", "expected"),
    [
        (10, 0.2, 2),
        (10, 0.2083, 2),
        (15, 0.2, 3),
        (24, 0.2, 5),
        (25, 0.2, 5),
        (30, 0.2, 6),
        (50, 0.2, 10),
        (60, 0.2, 12),
        (10, 0.05, 1),  # below one frame: clamp
        (25, 0.04, 1),
        (25, 0.1, 3),  # 2.5 rounds half-up
        (24, 0.1, 2),
    ],
)
def test_gap_for(fps, dt, expected):
    assert gap_for(fps, dt) == expected


def test_rounding_and_conversions():
    assert round_half_up(2.5) == 3
    assert round_half_up(2.49) == 2
    assert effective_dt(10, 2) == pytest.approx(0.2)
    assert seconds_to_frames(1.5, 10) == 15
    assert seconds_to_frames(0.01, 10) == 0
    assert frames_to_seconds(15, 10) == pytest.approx(1.5)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        gap_for(0, 0.2)
    with pytest.raises(ValueError):
        gap_for(10, 0)
    with pytest.raises(ValueError):
        effective_dt(10, 0)


def test_plan_at_source_rate():
    samples = plan_samples(101, 10.0, 0.2, emulate_fps=10.0)
    assert len(samples) == 99
    assert samples[0].current_index == 2
    assert samples[-1].current_index == 100
    assert all(sample.source_gap == 2 for sample in samples)
    assert all(sample.effective_dt == pytest.approx(0.2) for sample in samples)
    assert not any(sample.duplicate for sample in samples)


def test_plan_upsampled_30fps_dedupes_to_source_frames():
    samples = plan_samples(101, 10.0, 0.2, emulate_fps=30.0)
    current = {sample.current_index for sample in samples}
    assert current == set(range(2, 101))
    assert len(samples) == 99
    assert all(sample.effective_dt == pytest.approx(0.2) for sample in samples)


def test_plan_downsampled_5fps_keeps_effective_window():
    samples = plan_samples(101, 10.0, 0.2, emulate_fps=5.0)
    assert samples
    assert all(sample.source_gap == 2 for sample in samples)
    assert all(sample.effective_dt == pytest.approx(0.2) for sample in samples)
    assert all(sample.current_index % 2 == 0 for sample in samples)
    assert samples[0].current_index == 2
    assert samples[-1].current_index == 100


def test_short_window_is_quantised_and_reported():
    samples = plan_samples(11, 10.0, 0.05, emulate_fps=10.0)
    assert samples
    assert all(sample.source_gap == 1 for sample in samples)
    assert all(sample.effective_dt == pytest.approx(0.1) for sample in samples)
    summary = summarize_plan(samples, 0.05)
    assert summary["nominal_dt"] == 0.05
    assert summary["effective_dt_mean"] == pytest.approx(0.1)
    assert summary["effective_dt_min"] == pytest.approx(0.1)


def test_summary_reports_repeated_source_frames():
    samples = plan_samples(
        101, 10.0, 0.2, emulate_fps=30.0, dedupe=False
    )
    summary = summarize_plan(samples, 0.2)
    assert summary["samples"] > summary["unique_source_frames"]
    assert summary["repeated_source_frames"] == (
        summary["samples"] - summary["unique_source_frames"]
    )
    assert summary["self_pairs"] == 0


def test_process_hz_decouples_cadence():
    samples = plan_samples(101, 10.0, 0.2, emulate_fps=10.0, process_hz=5.0)
    assert len(samples) == 50
    assert plan_cadence_hz(samples, 10.0) == pytest.approx(5.0, rel=1e-3)


def test_min_source_index_and_ranges():
    samples = plan_samples(101, 10.0, 0.2, emulate_fps=10.0, min_source_index=5)
    assert samples[0].current_index == 5
    ranged = plan_samples(
        101, 10.0, 0.2, emulate_fps=10.0, start_s=1.0, end_s=2.0
    )
    assert ranged[0].current_index == 10
    assert ranged[-1].current_index == 20
