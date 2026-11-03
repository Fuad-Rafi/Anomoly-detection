"""Replace AI-VAD's min-max score normalization with robust percentiles.

Anomalib normalizes each density estimator with the minimum and maximum score seen
during fitting. A handful of outlier training pairs (camera cuts, decode glitches)
can push the maximum far above the normal range; the velocity estimator then maps
both walking and fighting to almost zero, which makes the motion signal unusable.

This script samples normal training clips, recomputes raw estimator scores, and
patches the saved checkpoint's normalization buffers to robust percentiles.

Examples:
    .\\.venv\\Scripts\\python.exe src\\recalibrate_aivad_norm.py `
        --checkpoint results/aivad/scene01_gap5/AiVad/ShanghaiTech/v0/weights/lightning/model.ckpt `
        --scene 1 --gap 5 --batches 80
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from anomalib.data import ShanghaiTech
from anomalib.models import AiVad
from anomalib.models.video.ai_vad.features import FeatureType

from fit_aivad_gap import gap_indexer_factory

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--scene", type=int, default=1)
    parser.add_argument("--gap", type=int, default=5)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--batches", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--low-percentile", type=float, default=1.0)
    parser.add_argument("--high-percentile", type=float, default=99.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.checkpoint.with_name("model_robust.ckpt")

    model = AiVad.load_from_checkpoint(
        str(args.checkpoint), map_location="cpu", post_processor=False
    )
    model.model.eval().to("cuda")
    core = model.model
    estimator = core.density_estimator

    datamodule = ShanghaiTech(
        root="data/shanghaitech",
        scene=args.scene,
        clip_length_in_frames=2,
        frames_between_clips=args.stride,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=0,
    )
    datamodule.setup()
    datamodule.train_data.indexer_cls = gap_indexer_factory(args.gap)
    datamodule.train_data._setup_clips()
    loader = datamodule.train_dataloader()

    raw = {"velocity": [], "deep": [], "pose": []}
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.batches:
                break
            clip = batch.image.to("cuda")
            first, last = clip[:, 0], clip[:, -1]
            flow = core.flow_extractor(first, last)
            regions = core.region_extractor(first, last)
            features = core.feature_extractor(first, flow, regions)

            for feature_set in features:
                if (
                    estimator.use_velocity_features
                    and feature_set[FeatureType.VELOCITY].numel()
                ):
                    raw["velocity"].append(
                        estimator.velocity_estimator.predict(
                            feature_set[FeatureType.VELOCITY], normalize=False
                        ).cpu()
                    )
                if estimator.use_deep_features and feature_set[FeatureType.DEEP].numel():
                    raw["deep"].append(
                        estimator.appearance_estimator.predict(
                            feature_set[FeatureType.DEEP], normalize=False
                        ).cpu()
                    )
                if estimator.use_pose_features and feature_set[FeatureType.POSE].numel():
                    raw["pose"].append(
                        estimator.pose_estimator.predict(
                            feature_set[FeatureType.POSE], normalize=False
                        ).cpu()
                    )
            if batch_index % 10 == 0:
                print(f"  batch {batch_index}/{args.batches}", flush=True)

    if not raw["velocity"]:
        raise SystemExit("No velocity scores collected")

    stats = {}
    raw_dump = {}
    for name, chunks in raw.items():
        if not chunks:
            continue
        values = torch.cat(chunks).numpy()
        raw_dump[name] = values
        low = float(np.percentile(values, args.low_percentile))
        high = float(np.percentile(values, args.high_percentile))
        stats[name] = (low, high)
        percentiles = {
            p: float(np.percentile(values, p)) for p in (50, 90, 95, 99, 99.5)
        }
        print(
            f"{name:9s} n={len(values):7d} min={values.min():8.3f} "
            f"median={percentiles[50]:8.3f} p90={percentiles[90]:8.3f} "
            f"p95={percentiles[95]:8.3f} p99={percentiles[99]:8.3f} "
            f"p99.5={percentiles[99.5]:8.3f} max={values.max():8.3f}"
        )
        print(
            f"          chosen p{args.low_percentile:g}={low:.3f} "
            f"p{args.high_percentile:g}={high:.3f}"
        )

    np.savez(output.with_name("raw_scores.npz"), **raw_dump)
    print(f"raw scores : {output.with_name('raw_scores.npz')}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    name_to_estimator = {
        "velocity": "velocity_estimator",
        "deep": "appearance_estimator",
        "pose": "pose_estimator",
    }
    for name, (low, high) in stats.items():
        prefix = f"model.density_estimator.{name_to_estimator[name]}"
        state[f"{prefix}.min"] = torch.tensor(low)
        state[f"{prefix}.max"] = torch.tensor(high)
        print(f"patched {prefix}: min={low:.3f} max={high:.3f}")

    torch.save(checkpoint, output)
    print(f"saved      : {output}")


if __name__ == "__main__":
    main()
