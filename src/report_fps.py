"""Render a Markdown summary of the Phase-1 frame-rate CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_summary(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    files = sorted((ROOT / "results/fps").glob("*.csv"))
    if not files:
        raise SystemExit("No results/fps/*.csv yet")

    lines = ["# Frame-rate robustness results", ""]
    for path in files:
        rows = read_summary(path)
        if not rows:
            continue
        lines.append(f"## {path.stem}")
        columns = [
            "run",
            "frames",
            "spearman",
            "pearson",
            "ks",
            "median_ratio",
            "alarm_jaccard",
            "box_jaccard",
            "ref_alarm_frames",
            "run_alarm_frames",
            "ref_box_frames",
            "run_box_frames",
            "event_matches",
            "first_alarm_delta_s",
            "run_auroc",
            "run_f1",
            "run_fa_per_min",
        ]
        present = [column for column in columns if column in rows[0]]
        lines.append("| " + " | ".join(present) + " |")
        lines.append("|" + "|".join("---" for _ in present) + "|")
        for row in rows:
            values = []
            for column in present:
                value = row.get(column, "")
                try:
                    number = float(value)
                    values.append(f"{number:.3f}")
                except (TypeError, ValueError):
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")

    output = ROOT / "results/fps/report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"written: {output}")


if __name__ == "__main__":
    main()
