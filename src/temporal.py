"""Time-based sampling utilities for frame-rate-robust video anomaly detection.

The detector is defined in *time*, not in frame counts.  A source at ``fps`` with a
temporal window ``dt`` uses ``gap = round(fps * dt)`` frames.  This module keeps the
rounding, the effective (achievable) window, and the frame-index mapping in one place
so that every experiment logs the same numbers.

Key concepts
------------
``target_dt``
    The physical window between the first and last frame of an AI-VAD clip, in seconds.
``source_fps``
    Native frames per second of the decoded stream.
``emulate_fps``
    Optional output cadence used to compare a source at another frame rate.  Values
    above ``source_fps`` duplicate source frames; values below it drop frames.  The
    effective window is always recomputed from the *source* indices.
``process_hz``
    Optional decision cadence.  Defaults to ``emulate_fps`` (process every framed
    sample).  Set it lower to keep compute constant when the camera frame rate rises.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def round_half_up(value: float) -> int:
    """Round a float to the nearest integer, with halves going away from zero."""
    return int(math.floor(float(value) + 0.5))


def gap_for(fps: float, dt: float) -> int:
    """Return the integer frame gap for a window ``dt`` at ``fps``.

    The result is clamped to at least one frame: a window shorter than one frame
    interval cannot be represented.  Callers should report ``effective_dt`` so the
    quantisation is visible rather than silently assumed.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    return max(1, round_half_up(fps * dt))


def effective_dt(fps: float, gap: int) -> float:
    """Physical window represented by ``gap`` frames at ``fps``."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if gap < 1:
        raise ValueError(f"gap must be >= 1, got {gap}")
    return gap / fps


def seconds_to_frames(seconds: float, fps: float) -> int:
    """Convert a duration to a frame count at ``fps`` (minimum zero)."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return max(0, round_half_up(float(seconds) * fps))


def frames_to_seconds(frames: float, fps: float) -> float:
    """Convert a frame count to seconds at ``fps``."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return float(frames) / fps


@dataclass(frozen=True)
class Sample:
    """One detector invocation planned on a source stream."""

    index: int
    time_s: float
    current_index: int
    first_index: int
    effective_dt: float
    duplicate: bool

    @property
    def source_gap(self) -> int:
        return self.current_index - self.first_index


def plan_samples(
    n_frames: int,
    source_fps: float,
    target_dt: float,
    *,
    emulate_fps: float | None = None,
    process_hz: float | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
    min_source_index: int = 1,
    dedupe: bool = True,
    allow_partial_start: bool = False,
) -> list[Sample]:
    """Plan detector samples on a stream.

    Args:
        n_frames: Number of source frames available.
        source_fps: Native frames per second.
        target_dt: Physical window between the two AI-VAD frames, in seconds.
        emulate_fps: Output cadence used for frame-rate experiments.  Defaults to
            ``source_fps``.
        process_hz: Decision cadence.  Defaults to ``emulate_fps``.
        start_s: Optional start time in seconds.
        end_s: Optional end time in seconds (inclusive).
        min_source_index: Do not emit samples whose current frame precedes this index.
            The legacy renderer starts at frame 1; video sources may start at 0.
        dedupe: Drop repeated ``(first_index, current_index)`` pairs.  This is what
            makes upsampled 20/30 fps runs cheap and equivalent to the source run.
        allow_partial_start: Emit samples whose ``t - target_dt`` falls before the
            start of the stream (the first window is then shorter).  The default is
            to skip them so every sample has the full temporal window.

    Returns:
        A list of :class:`Sample` ordered by source frame index.
    """
    if n_frames < 2:
        return []
    if source_fps <= 0:
        raise ValueError(f"source_fps must be positive, got {source_fps}")
    if target_dt <= 0:
        raise ValueError(f"target_dt must be positive, got {target_dt}")

    cadence = float(process_hz) if process_hz else float(emulate_fps or source_fps)
    if cadence <= 0:
        raise ValueError(f"cadence must be positive, got {cadence}")

    duration = (n_frames - 1) / source_fps
    start = 0.0 if start_s is None else max(0.0, float(start_s))
    end = duration if end_s is None else min(duration, float(end_s))
    if start > end + 1e-9:
        return []

    source_gap = gap_for(source_fps, target_dt)
    step_s = 1.0 / cadence
    # First grid point whose current source frame is at or after min_source_index.
    min_t = max(start, float(min_source_index) / source_fps)
    if not allow_partial_start:
        min_t = max(min_t, source_gap / source_fps)
    first_k = max(0, math.ceil((min_t - start) * cadence - 1e-9))

    samples: list[Sample] = []
    seen: set[tuple[int, int]] = set()
    k = first_k
    while True:
        t = start + k * step_s
        if t > end + 1e-9:
            break
        current = min(n_frames - 1, max(0, round_half_up(t * source_fps)))
        if not allow_partial_start and current < source_gap:
            k += 1
            continue
        first = max(0, current - source_gap)
        if current >= min_source_index:
            key = (first, current)
            if not dedupe or key not in seen:
                seen.add(key)
                samples.append(
                    Sample(
                        index=len(samples),
                        time_s=current / source_fps,
                        current_index=current,
                        first_index=first,
                        effective_dt=(current - first) / source_fps,
                        duplicate=first == current,
                    )
                )
        k += 1
        if k > 50_000_000:  # pragma: no cover - guards a malformed plan
            raise RuntimeError("sample planning did not terminate")

    samples.sort(key=lambda sample: (sample.current_index, sample.first_index))
    return samples


def plan_cadence_hz(samples: list[Sample], fallback: float) -> float:
    """Estimate the decision cadence (Hz) of a planned sample list."""
    if len(samples) < 2:
        return fallback
    times = np.asarray([sample.time_s for sample in samples], dtype=np.float64)
    diffs = np.diff(times)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return fallback
    return float(1.0 / np.median(diffs))


def summarize_plan(samples: list[Sample], nominal_dt: float) -> dict:
    """Return a JSON-serialisable summary of a sample plan."""
    if not samples:
        return {
            "samples": 0,
            "unique_source_frames": 0,
            "repeated_source_frames": 0,
            "self_pairs": 0,
            "nominal_dt": nominal_dt,
            "effective_dt_mean": None,
            "effective_dt_min": None,
            "effective_dt_max": None,
            "source_gap_min": 0,
            "source_gap_max": 0,
            "first_time_s": None,
            "last_time_s": None,
        }
    gaps = np.asarray([sample.source_gap for sample in samples], dtype=np.int64)
    dts = np.asarray([sample.effective_dt for sample in samples], dtype=np.float64)
    unique_current = len({sample.current_index for sample in samples})
    return {
        "samples": len(samples),
        "unique_source_frames": unique_current,
        "repeated_source_frames": len(samples) - unique_current,
        "self_pairs": int(sum(sample.duplicate for sample in samples)),
        "nominal_dt": nominal_dt,
        "effective_dt_mean": float(dts.mean()),
        "effective_dt_min": float(dts.min()),
        "effective_dt_max": float(dts.max()),
        "source_gap_min": int(gaps.min()),
        "source_gap_max": int(gaps.max()),
        "first_time_s": float(samples[0].time_s),
        "last_time_s": float(samples[-1].time_s),
    }


class FolderFrameSource:
    """Read frames from a folder of images, with a small LRU cache."""

    def __init__(self, directory: Path | str, fps: float, pattern: str = "*.jpg", cache_size: int = 256) -> None:
        self.directory = Path(directory)
        self.paths = sorted(self.directory.glob(pattern))
        if not self.paths:
            raise FileNotFoundError(f"No files matching {pattern!r} in {self.directory}")
        self.fps = float(fps)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = max(4, int(cache_size))

    def __len__(self) -> int:
        return len(self.paths)

    def get(self, index: int) -> np.ndarray:
        if index < 0 or index >= len(self.paths):
            raise IndexError(f"frame index {index} out of range [0, {len(self.paths) - 1}]")
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached
        image = cv2.imread(str(self.paths[index]))
        if image is None:
            raise RuntimeError(f"Could not read frame {self.paths[index]}")
        self._cache[index] = image
        self._cache.move_to_end(index)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return image

    def close(self) -> None:
        self._cache.clear()


class VideoFrameSource:
    """Sequential video decoder with random-access caching.

    OpenCV seeking is slow and unreliable on some codecs, so this class decodes
    forward and keeps the recent window.  Access is monotonic in practice (current
    frames and their ``current - gap`` partners), so a single cursor is enough.
    """

    def __init__(self, path: Path | str, fps: float | None = None, cache_size: int = 256) -> None:
        self.path = Path(path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video {self.path}")
        detected_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self.fps = float(fps) if fps else detected_fps
        if self.fps <= 0:
            raise RuntimeError(f"Could not determine fps for {self.path}; pass --source-fps")
        self._count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self._count <= 0:
            raise RuntimeError(f"Could not determine frame count for {self.path}")
        self._next = 0
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = max(4, int(cache_size))

    def __len__(self) -> int:
        return self._count

    def _reset(self) -> None:
        self._cap.release()
        self._cap = cv2.VideoCapture(str(self.path))
        self._next = 0
        self._cache.clear()

    def get(self, index: int) -> np.ndarray:
        if index < 0 or index >= self._count:
            raise IndexError(f"frame index {index} out of range [0, {self._count - 1}]")
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached
        if index < self._next:
            self._reset()
        while self._next <= index:
            ok, frame = self._cap.read()
            if not ok:
                raise RuntimeError(f"Decoding failed at frame {self._next} of {self.path}")
            self._cache[self._next] = frame
            self._next += 1
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return self._cache[index]

    def close(self) -> None:
        self._cap.release()
        self._cache.clear()
