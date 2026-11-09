"""Paper baseline: OT between raw pixel vectors (L1 ground cost), same solver interface.

    python -m baseline.raw_ot INSTANCES_DIR OUTPUT_DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import ot
from scipy.spatial.distance import cdist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("instances_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--metric", default="cityblock")
    parser.add_argument("--reg", type=float, default=0.01)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.instances_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            xs = z["source"].reshape(len(z["source"]), -1)
            xt = z["target"].reshape(len(z["target"]), -1)
        cost = cdist(xs, xt, metric=args.metric).astype(np.float64)
        cost /= max(float(cost.max()), 1e-300)
        a = np.full(len(xs), 1.0 / len(xs))
        b = np.full(len(xt), 1.0 / len(xt))
        plan = ot.sinkhorn(a, b, cost, reg=args.reg, method="greenkhorn", numItermax=1_000_000, stopThr=1e-9, warn=False)
        np.savez(args.output_dir / path.name, coupling=plan)


if __name__ == "__main__":
    main()
