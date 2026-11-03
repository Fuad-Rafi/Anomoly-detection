"""Fit AI-VAD with a deployment-matched within-clip frame gap.

Anomalib's `ShanghaiTech` datamodule samples clips of consecutive frames: its
`frames_between_clips` argument controls the stride *between* clips, not the gap
*inside* a clip. AI-VAD's velocity features are computed from the first and last
frame of a clip, so a checkpoint fitted on consecutive frames is out of
distribution when inference uses a 0.5 s (5-frame) gap.

This script swaps the training indexer for one that samples `(t, t + gap)` pairs,
then fits a normal AI-VAD checkpoint on them.

Examples:
    .\\.venv\\Scripts\\python.exe src\\fit_aivad_gap.py --scene 1 --gap 5 --stride 25
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from anomalib.data import ShanghaiTech
from anomalib.data.datasets.video.shanghaitech import ShanghaiTechTrainClipsIndexer
from anomalib.engine import Engine
from anomalib.models import AiVad
from torchcodec.decoders import VideoDecoder
from torchvision.datasets.video_utils import VideoClips

ROOT = Path(__file__).resolve().parent.parent


class GapTrainClipsIndexer(ShanghaiTechTrainClipsIndexer):
    """ShanghaiTech training indexer with an exact within-clip frame step.

    ``VideoClips.frame_rate`` is a sampling rate, so the resulting step depends on
    each video's native fps (the ShanghaiTech training videos are 24 fps). This
    indexer instead rewrites the frame index table so every clip is exactly
    ``(t, t + frame_step)``.
    """

    def __init__(
        self,
        video_paths: list[str],
        mask_paths: list[str],
        clip_length_in_frames: int = 2,
        frames_between_clips: int = 1,
        frame_step: int = 1,
    ) -> None:
        VideoClips.__init__(
            self,
            video_paths=video_paths,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            frame_rate=None,
            output_format="TCHW",
        )
        self.mask_paths = mask_paths

        if frame_step < 1:
            raise ValueError("frame_step must be >= 1")

        patched = []
        for clip_table in self.clips:
            if len(clip_table) == 0:
                patched.append(clip_table)
                continue
            starts = clip_table[:, 0]
            ends = starts + frame_step
            keep = ends < self.last_frame_idx(len(patched))
            starts = starts[keep]
            ends = ends[keep]
            patched.append(torch.stack([starts, ends], dim=1))
        self.clips = patched

        # Rebuild the flattened clip -> (video, clip) index after shortening clips.
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


def gap_indexer_factory(gap: int):
    def factory(
        video_paths: list[str],
        mask_paths: list[str],
        clip_length_in_frames: int = 2,
        frames_between_clips: int = 1,
    ) -> GapTrainClipsIndexer:
        return GapTrainClipsIndexer(
            video_paths=video_paths,
            mask_paths=mask_paths,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            frame_step=gap,
        )

    return factory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=int, default=1)
    parser.add_argument("--gap", type=int, default=5)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--root", default="data/shanghaitech")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(
        args.output_root
        or f"results/aivad/scene{args.scene:02d}_gap{args.gap}"
    )

    datamodule = ShanghaiTech(
        root=args.root,
        scene=args.scene,
        clip_length_in_frames=2,
        frames_between_clips=args.stride,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup()

    # Replace the training indexer with a gap-aware one.
    datamodule.train_data.indexer_cls = gap_indexer_factory(args.gap)
    datamodule.train_data._setup_clips()

    # Sanity check: the two frames in a clip must be `gap` apart.
    item = datamodule.train_data[0]
    frames = item.image
    print(
        f"scene {args.scene}: {len(datamodule.train_data)} training clips, "
        f"clip shape {tuple(frames.shape)}, gap {args.gap}, stride {args.stride}",
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
    print(
        f"FIT DONE scene={args.scene} gap={args.gap} "
        f"time={elapsed:.1f}s detections={model.total_detections}",
        flush=True,
    )


if __name__ == "__main__":
    main()
