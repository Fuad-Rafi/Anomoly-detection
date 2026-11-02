"""Fit AI-VAD with a deployment-matched temporal sampling gap.

The original checkpoint used a preparation stride of 100. Product experiments should
fit separate checkpoints with a small gap that matches the renderer and camera FPS.

Example:
    .\.venv\Scripts\python.exe src\fit_aivad.py --scene 1 --frame-gap 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from anomalib.data import ShanghaiTech
from anomalib.engine import Engine
from anomalib.models import AiVad

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=int, default=1)
    parser.add_argument("--frame-gap", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "aivad")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_gap < 1:
        raise SystemExit("--frame-gap must be at least 1")
    root = ROOT / "data" / "shanghaitech"
    datamodule = ShanghaiTech(
        root=str(root),
        scene=args.scene,
        clip_length_in_frames=2,
        frames_between_clips=args.frame_gap,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.workers,
    )
    model = AiVad(
        persons_only=True,
        enable_foreground_detections=False,
        pre_processor=False,
        post_processor=False,
    )
    engine = Engine(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=args.max_epochs,
        limit_val_batches=0,
        default_root_dir=str(args.output / f"scene{args.scene:02d}_gap{args.frame_gap}"),
    )
    engine.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
