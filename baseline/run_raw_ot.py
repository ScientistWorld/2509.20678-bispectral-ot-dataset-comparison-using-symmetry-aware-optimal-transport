"""Run the raw-feature OT baseline using the shared method artifact format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import ot
from scipy.spatial.distance import cdist

from data.idx import load_training_split
from data.preprocessing import class_balanced_halves, normalize_images, raw_features, rotate_images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "kmnist"])
    parser.add_argument("--data-root", default="data/downloads")
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rotation-seed", type=int, default=1)
    parser.add_argument("--reg", type=float, default=0.01)
    parser.add_argument("--out-dir", default="outputs/baseline")
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
    manifest: dict[str, object] = {
        "dataset": args.dataset,
        "samples_per_class": args.samples_per_class,
        "seed": args.seed,
        "rotation_seed": args.rotation_seed,
        "reg": args.reg,
        "experiments": {},
    }
    for exp_name, src_imgs in {
        f"{args.dataset}_rotated_matching": rotated_source,
        f"{args.dataset}_unrotated_baseline": source_images,
    }.items():
        src_features = raw_features(src_imgs)
        tgt_features = raw_features(target_images)
        cost = cdist(src_features, tgt_features, metric="cityblock").astype(np.float64)
        cost /= max(float(cost.max()), 1e-12)
        a = np.full(len(source_labels), 1.0 / len(source_labels), dtype=np.float64)
        b = np.full(len(target_labels), 1.0 / len(target_labels), dtype=np.float64)
        gamma = ot.sinkhorn(a, b, cost, reg=args.reg, method="greenkhorn", numItermax=1_000_000)
        artifact = out_dir / f"{exp_name}__raw_ot_l1.npz"
        np.savez_compressed(
            artifact,
            gamma=gamma,
            source_labels=source_labels,
            target_labels=target_labels,
            rotation_angles=angles if "rotated" in exp_name else np.zeros_like(angles),
        )
        manifest["experiments"].setdefault(exp_name, {})["raw_ot_l1"] = str(artifact)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
