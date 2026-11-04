"""Fit AI-VAD with a deployment-matched within-clip frame gap.

Anomalib's video datamodules sample clips of consecutive frames: their
``frames_between_clips`` argument controls the stride *between* clips, not the gap
*inside* a clip.  AI-VAD's velocity features are computed from the first and last
frame, so a checkpoint fitted on consecutive frames is out of distribution when
inference uses a larger temporal window.

This script swaps the training indexer for one that samples exact ``(t, t + gap)``
pairs, then fits a normal AI-VAD checkpoint.  Both ShanghaiTech and Avenue are
supported because the temporal gap must be matched on every dataset.

Examples:
    .\\.venv\\Scripts\\python.exe src\\fit_aivad_gap.py --scene 1 --gap 5 --stride 25
    .\\.venv\\Scripts\\python.exe src\\fit_aivad_gap.py --dataset avenue --gap 5 --stride 25
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from anomalib.data import Avenue, ShanghaiTech
from anomalib.data.datasets.video.avenue import AvenueClipsIndexer
from anomalib.data.datasets.video.shanghaitech import ShanghaiTechTrainClipsIndexer
from anomalib.engine import Engine
from anomalib.models import AiVad
from torchcodec.decoders import VideoDecoder

ROOT = Path(__file__).resolve().parent.parent


class ExactGapIndexerMixin:
    """Replace a ``VideoClips`` index table with exact ``(t, t + gap)`` pairs.

    ``VideoClips.frame_rate`` is a sampling rate, so the resulting step depends on
    each video's native fps.  Patching the index table makes the gap explicit and
    independent of the source frame rate.
    """

    def _apply_gap(self, frame_step: int) -> None:
        if frame_step < 1:
            raise ValueError("frame_step must be >= 1")

        # Cache the last frame index before replacing the tables, because
        # ``last_frame_idx`` reads from ``self.clips``.
        last_frames = [
            int(table[-1][-1].item()) if len(table) else -1 for table in self.clips
        ]

        patched = []
        for video_idx, clip_table in enumerate(self.clips):
            if len(clip_table) == 0:
                patched.append(clip_table)
                continue
            starts = clip_table[:, 0]
            ends = starts + frame_step
            keep = ends <= last_frames[video_idx]
            starts = starts[keep]
            ends = ends[keep]
            patched.append(torch.stack([starts, ends], dim=1))
        self.clips = patched

        self.cumulative_sizes = []
        total = 0
        for clip_table in self.clips:
            total += len(clip_table)
            self.cumulative_sizes.append(total)

    def get_clip(self, idx: int):
        """Decode exactly the two frames selected for this clip."""
        if idx >= self.num_clips():
            raise IndexError(f"Index {idx} out of range ({self.num_clips()} clips)")
        video_idx, clip_idx = self.get_clip_location(idx)
        video_path = self.video_paths[video_idx]
        clip_pts = self.clips[video_idx][clip_idx]
        start_idx = int(clip_pts[0].item())
        end_idx = int(clip_pts[-1].item())

        decoder = VideoDecoder(video_path, dimension_order="NCHW")
        video = decoder.get_frames_at(indices=[start_idx, end_idx]).data
        info = {"video_fps": decoder.metadata.average_fps}
        audio = torch.empty((1, 0), dtype=torch.float32)
        return video, audio, info, video_idx


class GapShanghaiTechIndexer(ExactGapIndexerMixin, ShanghaiTechTrainClipsIndexer):
    """ShanghaiTech training indexer with an exact within-clip frame step."""

    def __init__(
        self,
        video_paths: list[str],
        mask_paths: list[str],
        clip_length_in_frames: int = 2,
        frames_between_clips: int = 1,
        frame_step: int = 1,
    ) -> None:
        ShanghaiTechTrainClipsIndexer.__init__(
            self,
            video_paths=video_paths,
            mask_paths=mask_paths,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
        )
        self._apply_gap(frame_step)


class GapAvenueIndexer(ExactGapIndexerMixin, AvenueClipsIndexer):
    """Avenue training indexer with an exact within-clip frame step."""

    def __init__(
        self,
        video_paths: list[str],
        mask_paths: list[str],
        clip_length_in_frames: int = 2,
        frames_between_clips: int = 1,
        frame_step: int = 1,
    ) -> None:
        AvenueClipsIndexer.__init__(
            self,
            video_paths=video_paths,
            mask_paths=mask_paths,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
        )
        self._apply_gap(frame_step)


def shanghaitech_gap_indexer_factory(gap: int):
    def factory(
        video_paths: list[str],
        mask_paths: list[str],
        clip_length_in_frames: int = 2,
        frames_between_clips: int = 1,
    ) -> GapShanghaiTechIndexer:
        return GapShanghaiTechIndexer(
            video_paths=video_paths,
            mask_paths=mask_paths,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            frame_step=gap,
        )

    return factory


def avenue_gap_indexer_factory(gap: int):
    def factory(
        video_paths: list[str],
        mask_paths: list[str],
        clip_length_in_frames: int = 2,
        frames_between_clips: int = 1,
    ) -> GapAvenueIndexer:
        return GapAvenueIndexer(
            video_paths=video_paths,
            mask_paths=mask_paths,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            frame_step=gap,
        )

    return factory


def build_datamodule(args: argparse.Namespace):
    if args.dataset == "shanghaitech":
        datamodule = ShanghaiTech(
            root=args.root or "data/shanghaitech",
            scene=args.scene,
            clip_length_in_frames=2,
            frames_between_clips=args.stride,
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        datamodule.setup()
        datamodule.train_data.indexer_cls = shanghaitech_gap_indexer_factory(args.gap)
        datamodule.train_data._setup_clips()
        return datamodule, (args.train_fps or 24.0)

    datamodule = Avenue(
        root=args.root or "data/avenue",
        gt_dir=args.gt_dir or "data/avenue/ground_truth_demo",
        clip_length_in_frames=2,
        frames_between_clips=args.stride,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup()
    datamodule.train_data.indexer_cls = avenue_gap_indexer_factory(args.gap)
    datamodule.train_data._setup_clips()
    return datamodule, (args.train_fps or 25.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["shanghaitech", "avenue"], default="shanghaitech")
    parser.add_argument("--scene", type=int, default=1)
    parser.add_argument("--gap", type=int, default=5)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--root", default=None)
    parser.add_argument("--gt-dir", default=None)
    parser.add_argument("--train-fps", type=float, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root:
        output_root = Path(args.output_root)
    elif args.dataset == "shanghaitech":
        output_root = Path(f"results/aivad/scene{args.scene:02d}_gap{args.gap}")
    else:
        output_root = Path(f"results/aivad/avenue_gap{args.gap}")
    output_root.mkdir(parents=True, exist_ok=True)

    datamodule, train_fps = build_datamodule(args)
    item = datamodule.train_data[0]
    frames = item.image
    training_clips = int(datamodule.train_data.indexer.num_clips())
    print(
        f"{args.dataset}: {training_clips} training clips, "
        f"clip shape {tuple(frames.shape)}, gap {args.gap}, stride {args.stride}, "
        f"effective dt {args.gap / train_fps:.4f}s",
        flush=True,
    )

    model = AiVad(
        persons_only=True,
        enable_foreground_detections=False,
        pre_processor=False,
        post_processor=False,
    )
    engine = Engine(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        default_root_dir=str(output_root),
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )
    if args.limit_train_batches:
        engine.limit_train_batches = args.limit_train_batches

    started = time.time()
    engine.fit(model=model, datamodule=datamodule)
    elapsed = time.time() - started

    metadata = {
        "dataset": args.dataset,
        "scene": args.scene if args.dataset == "shanghaitech" else None,
        "root": str(args.root or (f"data/{args.dataset}")),
        "train_fps": train_fps,
        "gap": args.gap,
        "effective_dt": args.gap / train_fps,
        "stride": args.stride,
        "clip_length_in_frames": 2,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "training_clips": training_clips,
        "detections": int(getattr(model, "total_detections", 0)),
        "elapsed_s": elapsed,
        "output_root": str(output_root),
    }
    (output_root / "fit_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"FIT DONE dataset={args.dataset} gap={args.gap} "
        f"time={elapsed:.1f}s detections={metadata['detections']}",
        flush=True,
    )
    print(f"metadata   : {output_root / 'fit_metadata.json'}", flush=True)


if __name__ == "__main__":
    main()
