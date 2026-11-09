"""Shortcut solvers replayed against the hardened evaluators.

    python -m redteam.hardening.shortcuts MODE INSTANCES_DIR OUTPUT_DIR

These reproduce, under the new solver interface, every shortcut REDTEAM.md
ran (plus two new ones). None of them reads eval/test/.

MODE
  uniform        constant coupling (valid transport plan, no information)
  zero           all-zero coupling
  nan            NaN coupling
  signed         uniform marginals but large negative/positive entries
  tiny           a 10x10 coupling instead of n x m ("tiny test")
  empty          writes nothing
  order_blocks   assumes rows are grouped by class (index // per-class count)
  forged_extras  uniform coupling + forged label/score arrays, and overwrites
                 scoring/scores_*.json and status files with perfect values
  nn_assignment  every source row sends all its mass to its nearest target
                 (BOT cost) - not a transport plan
  train_lookup   looks every image up in the public training split; when all
                 labels are found it returns the label-block coupling
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.idx import load_training_split
from data.preprocessing import normalize_images
from method.bot_features import polar_bispectrum_features
from method.solve import cost_matrix, iterate_instances, save_coupling

NUM_CLASSES = {"mnist": 10, "kmnist": 10, "fmnist": 10, "emnist": 26}
_LOOKUP: dict[str, dict[bytes, int]] = {}


def lookup_table(dataset: str) -> dict[bytes, int]:
    if dataset not in _LOOKUP:
        images, labels = load_training_split("data/downloads", dataset)
        normed = normalize_images(images, dataset)
        _LOOKUP[dataset] = {img.tobytes(): int(y) for img, y in zip(normed, labels)}
    return _LOOKUP[dataset]


def block_coupling(ys: np.ndarray, yt: np.ndarray) -> np.ndarray:
    g = (ys[:, None] == yt[None, :]).astype(np.float64)
    return g / g.sum()


def solve(mode: str, source: np.ndarray, target: np.ndarray, dataset: str, out_dir: Path, iid: str) -> np.ndarray | None:
    n, m = len(source), len(target)
    if mode == "uniform":
        return np.full((n, m), 1.0 / (n * m))
    if mode == "zero":
        return np.zeros((n, m))
    if mode == "nan":
        return np.full((n, m), np.nan)
    if mode == "signed":
        u = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
        v = np.where(np.arange(m) % 2 == 0, 1.0, -1.0)
        return 1.0 / (n * m) + 1e6 * np.outer(u, v) / (n * m)
    if mode == "tiny":
        return np.full((10, 10), 0.01)
    if mode == "empty":
        return None
    if mode == "order_blocks":
        k = NUM_CLASSES[dataset]
        return block_coupling(np.arange(n) * k // n, np.arange(m) * k // m)
    if mode == "forged_extras":
        np.savez(
            out_dir / f"{iid}.npz",
            coupling=np.full((n, m), 1.0 / (n * m)),
            source_labels=np.zeros(n, dtype=np.int64),
            target_labels=np.zeros(m, dtype=np.int64),
            class_preservation_accuracy=np.float64(1.0),
            valid=np.int64(1),
        )
        perfect = {"class_preservation_accuracy": 1.0, "same_class_mass_mean": 1.0, "class_mass_margin_mean": 1.0}
        exps = [f"{d}_{c}" for d in NUM_CLASSES for c in ("rotated_matching", "unrotated_baseline")]
        for name in ("scores_test.json", "scores_train.json", "scores.json"):
            Path("scoring", name).write_text(json.dumps({"experiments": {e: {"results": {"method": perfect}} for e in exps}}))
        for name in ("test_status.json", "train_status.json"):
            Path("scoring", name).write_text(json.dumps({"status": "ok", "failures": []}))
        return "written"  # type: ignore[return-value]
    if mode == "nn_assignment":
        feats = lambda x: polar_bispectrum_features(x, 40)  # noqa: E731
        cost = cost_matrix(source, target, feats, "cityblock", "bispectral-nearest-40")
        g = np.zeros((n, m))
        g[np.arange(n), cost.argmin(axis=1)] = 1.0 / n
        return g
    if mode == "train_lookup":
        table = lookup_table(dataset)
        ys = np.array([table.get(x.tobytes(), -1) for x in source])
        yt = np.array([table.get(x.tobytes(), -1) for x in target])
        print(f"[shortcut:train_lookup] found {np.mean(ys >= 0):.3f} source / {np.mean(yt >= 0):.3f} target labels", flush=True)
        if (ys >= 0).all() and (yt >= 0).all():
            return block_coupling(ys, yt)
        return np.full((n, m), 1.0 / (n * m))
    raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("instances_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for iid, source, target, dataset in iterate_instances(args.instances_dir):
        g = solve(args.mode, source, target, dataset, args.output_dir, iid)
        if g is None or isinstance(g, str):
            continue
        save_coupling(args.output_dir, iid, g)


if __name__ == "__main__":
    main()
