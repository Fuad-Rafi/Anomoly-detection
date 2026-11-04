"""Compare a refactored renderer run against the saved legacy CSV.

Used as a guard during the aivad_core extraction.  Numeric columns that must not
change are compared; JSON diagnostic columns are ignored because their ordering can
differ without changing the decision.

Example:
    .\\.venv\\Scripts\\python.exe src\\compare_golden.py ^
        --legacy results/fps/golden/legacy.scores.csv ^
        --new results/fps/golden/new.scores.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

NUMERIC_COLUMNS = (
    "score",
    "raw_score",
    "map_max",
    "velocity_max",
    "pose_max",
    "deep_max",
    "boxes",
)


def load(path: Path) -> dict[int, dict]:
    with path.open(encoding="utf-8") as handle:
        return {int(row["frame_index"]): row for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    legacy = load(args.legacy)
    new = load(args.new)
    failures = []

    if set(legacy) != set(new):
        missing = sorted(set(legacy) - set(new))[:5]
        extra = sorted(set(new) - set(legacy))[:5]
        failures.append(f"frame sets differ: missing={missing} extra={extra}")

    for frame in sorted(set(legacy) & set(new)):
        left, right = legacy[frame], new[frame]
        for column in NUMERIC_COLUMNS:
            if column not in left or column not in right:
                continue
            delta = abs(float(left[column]) - float(right[column]))
            if delta > args.tolerance:
                failures.append(
                    f"frame {frame} column {column}: {left[column]} vs "
                    f"{right[column]} (delta {delta:.6f})"
                )

    if failures:
        print("GOLDEN FAIL")
        for failure in failures[:20]:
            print(f"  - {failure}")
        print(f"  {len(failures)} mismatches (tolerance {args.tolerance})")
        sys.exit(1)

    print(
        f"GOLDEN PASS: {len(legacy)} frames, columns={NUMERIC_COLUMNS}, "
        f"tolerance={args.tolerance}"
    )


if __name__ == "__main__":
    main()
