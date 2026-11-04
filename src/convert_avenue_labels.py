"""Convert CUHK Avenue ``*_label.mat`` masks to frame-level ``.npy`` labels.

Avenue stores one binary mask per frame in a MATLAB cell array.  For frame-level
precision/recall we only need a 0/1 label per frame: 1 if any anomaly pixel is set.

Examples:
    .\\.venv\\Scripts\\python.exe src\\convert_avenue_labels.py `
        --input data\\avenue\\ground_truth_demo\\testing_label_mask `
        --output data\\avenue\\testing_labels_npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio


def convert_file(source: Path, destination: Path) -> tuple[int, int]:
    mat = sio.loadmat(source)
    if "volLabel" not in mat:
        raise SystemExit(f"volLabel not found in {source}")
    frames = mat["volLabel"].ravel()
    labels = np.asarray(
        [int(np.asarray(frame).sum() > 0) for frame in frames], dtype=np.int8
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, labels)
    return len(labels), int(labels.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.input.is_dir():
        files = sorted(args.input.glob("*_label.mat"))
        if not files:
            raise SystemExit(f"No *_label.mat files in {args.input}")
        for source in files:
            stem = source.name.removesuffix("_label.mat")
            destination = args.output / f"{int(stem):02d}.npy" if stem.isdigit() else args.output / f"{stem}.npy"
            total, positives = convert_file(source, destination)
            print(f"{source.name} -> {destination.name}: frames={total} positives={positives}")
    else:
        total, positives = convert_file(args.input, args.output)
        print(f"{args.input} -> {args.output}: frames={total} positives={positives}")


if __name__ == "__main__":
    main()
