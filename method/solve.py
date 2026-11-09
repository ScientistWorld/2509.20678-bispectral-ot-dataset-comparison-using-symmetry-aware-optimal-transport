"""Bispectral OT solver (the paper's method) behind the gym's solver interface.

    python -m method.solve INSTANCES_DIR OUTPUT_DIR [--metric cityblock] [--reg 0.01]

For every ``INSTANCES_DIR/<id>.npz`` (arrays ``source`` (n, H, W), ``target``
(m, H, W), ``dataset``) this writes ``OUTPUT_DIR/<id>.npz`` holding
``coupling`` (n, m): the entropic OT plan between uniform measures on the two
image sets, with the ground cost taken between Z/KZ-bispectral polar features
(paper Sec. 3; K = 40 angles, R = min(H, W) // 2 radii), normalised by its
maximum, and solved with POT's Greenkhorn at regularisation 0.01.

Set ``BOT_COST_CACHE=<dir>`` to memoise cost matrices by instance content
(used by the hardening audit to share costs across variants; results are
identical with or without it).
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Callable

import numpy as np
import ot
from scipy.spatial.distance import cdist

from method.bot_features import polar_bispectrum_features

FeatureFn = Callable[[np.ndarray], np.ndarray]


def raw_features(images: np.ndarray) -> np.ndarray:
    return images.reshape(len(images), -1).astype(np.float32)


def cost_matrix(source: np.ndarray, target: np.ndarray, feature_fn: FeatureFn, metric: str, tag: str) -> np.ndarray:
    cache_dir = os.environ.get("BOT_COST_CACHE")
    path = None
    if cache_dir:
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(source, dtype=np.float32).tobytes())
        digest.update(np.ascontiguousarray(target, dtype=np.float32).tobytes())
        digest.update(f"|{tag}|{metric}|{source.shape}|{target.shape}".encode())
        path = Path(cache_dir) / f"{digest.hexdigest()[:40]}.npy"
        if path.exists():
            return np.load(path)
    cost = cdist(feature_fn(source), feature_fn(target), metric=metric).astype(np.float64)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp.npy")
        np.save(tmp, cost)
        os.replace(tmp, path)
    return cost


def entropic_plan(cost: np.ndarray, reg: float) -> np.ndarray:
    cost = cost.copy()
    max_cost = float(cost.max())
    if max_cost > 0:
        cost /= max_cost
    a = np.full(cost.shape[0], 1.0 / cost.shape[0])
    b = np.full(cost.shape[1], 1.0 / cost.shape[1])
    return ot.sinkhorn(a, b, cost, reg=reg, method="greenkhorn", numItermax=1_000_000, stopThr=1e-9, warn=True)


def bispectral_ot(source: np.ndarray, target: np.ndarray, metric: str = "cityblock", reg: float = 0.01, angles: int = 40) -> np.ndarray:
    features = lambda x: polar_bispectrum_features(x, angles)  # noqa: E731
    return entropic_plan(cost_matrix(source, target, features, metric, f"bispectral-nearest-{angles}"), reg)


def iterate_instances(instances_dir: Path):
    for path in sorted(instances_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            yield path.stem, z["source"], z["target"], str(z["dataset"])


def save_coupling(out_dir: Path, instance_id: str, coupling: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{instance_id}.npz", coupling=coupling)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("instances_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--metric", default="cityblock", choices=["cityblock", "euclidean", "sqeuclidean", "cosine"])
    parser.add_argument("--reg", type=float, default=0.01)
    parser.add_argument("--angles", type=int, default=40)
    args = parser.parse_args()
    for instance_id, source, target, dataset in iterate_instances(args.instances_dir):
        coupling = bispectral_ot(source, target, args.metric, args.reg, args.angles)
        save_coupling(args.output_dir, instance_id, coupling)
        print(f"[method.solve] {instance_id} ({dataset}) {source.shape[0]}x{target.shape[0]}", flush=True)


if __name__ == "__main__":
    main()
