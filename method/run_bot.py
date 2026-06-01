"""Run scaled Bispectral OT experiments and write reusable artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import ot
from scipy.spatial.distance import cdist

from data.idx import load_training_split
from data.preprocessing import (
    class_balanced_halves,
    normalize_images,
    raw_features,
    rotate_images,
)
from method.bot_features import polar_bispectrum_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "kmnist"])
    parser.add_argument("--data-root", default="data/downloads")
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rotation-seed", type=int, default=1)
    parser.add_argument("--angles", type=int, default=40)
    parser.add_argument("--reg", type=float, default=0.01)
    parser.add_argument("--metric", default="cityblock", choices=["cityblock", "euclidean", "sqeuclidean", "cosine"])
    parser.add_argument("--out-dir", default="outputs/bot")
    args = parser.parse_args()

    images, labels = load_training_split(args.data_root, args.dataset)
    images = normalize_images(images, args.dataset)
    source_idx, target_idx = class_balanced_halves(labels, args.samples_per_class, args.seed)

    source_images = images[source_idx]
    target_images = images[target_idx]
    source_labels = labels[source_idx].astype(int)
    target_labels = labels[target_idx].astype(int)
    rotated_source, angles = rotate_images(source_images, args.rotation_seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = {
        f"{args.dataset}_rotated_matching": (rotated_source, target_images),
        f"{args.dataset}_unrotated_baseline": (source_images, target_images),
    }
    methods = {
        "raw_ot_l1": raw_features,
        "bispectral_ot_l1": lambda x: polar_bispectrum_features(x, args.angles),
    }

    manifest: dict[str, object] = {
        "dataset": args.dataset,
        "samples_per_class": args.samples_per_class,
        "seed": args.seed,
        "rotation_seed": args.rotation_seed,
        "angles": args.angles,
        "reg": args.reg,
        "metric": args.metric,
        "experiments": {},
    }

    for exp_name, (src_imgs, tgt_imgs) in experiments.items():
        manifest["experiments"][exp_name] = {}
        for method_name, feature_fn in methods.items():
            src_features = feature_fn(src_imgs)
            tgt_features = feature_fn(tgt_imgs)
            cost = cdist(src_features, tgt_features, metric=args.metric).astype(np.float64)
            max_cost = float(cost.max())
            if max_cost > 0:
                cost /= max_cost
            a = np.full(len(source_labels), 1.0 / len(source_labels), dtype=np.float64)
            b = np.full(len(target_labels), 1.0 / len(target_labels), dtype=np.float64)
            gamma = ot.sinkhorn(
                a,
                b,
                cost,
                reg=args.reg,
                method="greenkhorn",
                numItermax=1_000_000,
                stopThr=1e-9,
                warn=True,
            )
            artifact = out_dir / f"{exp_name}__{method_name}.npz"
            np.savez_compressed(
                artifact,
                gamma=gamma,
                source_labels=source_labels,
                target_labels=target_labels,
                rotation_angles=angles if "rotated" in exp_name else np.zeros_like(angles),
                cost_max=max_cost,
            )
            manifest["experiments"][exp_name][method_name] = str(artifact)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
