"""Honest alternative solvers used to check that the hardened test rewards better science.

    python -m redteam.hardening.variants MODE INSTANCES_DIR OUTPUT_DIR

MODE
  bilinear      BOT with polar sampling centred on the rotation centre and bilinear
                interpolation (same bispectrum, cost, solver).                 [better]
  adaptive      Detects from the unlabeled images whether source and target differ by a
                nuisance transformation (nearest-neighbour distance ratio in pixel space);
                uses raw-pixel OT when they do not, BOT otherwise.             [better]
  combined      adaptive + bilinear.                                           [better]
  raw_geometry  Raw-pixel OT for every instance (removes the bispectrum).      [weaker]
  lower_reg     BOT with entropic regularisation 0.003 instead of 0.01.        [trial]
  bot_l22       BOT with the paper's squared-L2 ground cost.                   [paper variant]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from method.bot_features import polar_bispectrum_features
from method.solve import cost_matrix, entropic_plan, iterate_instances, raw_features, save_coupling

ANGLES = 40
SHIFT_THRESHOLD = 1.1


def bilinear_polar_bispectrum(images: np.ndarray, num_angles: int = ANGLES, chunk: int = 256) -> np.ndarray:
    n, h, w = images.shape
    radii = np.arange(min(h, w) // 2, dtype=np.float64)
    theta = 2.0 * np.pi * np.arange(num_angles) / num_angles
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ys = cy + radii[:, None] * np.sin(theta)[None, :]
    xs = cx + radii[:, None] * np.cos(theta)[None, :]
    coords = np.stack([ys.ravel(), xs.ravel()])
    idx = (np.arange(num_angles)[:, None] + np.arange(num_angles)[None, :]) % num_angles
    out = []
    for start in range(0, n, chunk):
        block = images[start : start + chunk].astype(np.float64)
        polar = np.stack(
            [ndimage.map_coordinates(img, coords, order=1, mode="constant", cval=0.0).reshape(len(radii), num_angles) for img in block]
        ) * radii[None, :, None]
        fft = np.fft.fft(polar, axis=2)
        bispec = fft[:, :, :, None] * fft[:, :, None, :] * np.conj(fft[:, :, idx])
        norms = np.linalg.norm(bispec.reshape(len(block), -1), axis=1)
        bispec = bispec / np.where(norms > 0, norms, 1.0)[:, None, None, None]
        flat = bispec.reshape(len(block), -1)
        out.append(np.concatenate([flat.real, flat.imag], axis=1).astype(np.float32))
    return np.concatenate(out)


def sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
    return np.maximum(d, 0.0)


def shift_statistic(source: np.ndarray, target: np.ndarray) -> float:
    """Median source->target NN distance over median target->target (leave-one-out) NN distance."""
    xs, xt = raw_features(source), raw_features(target)
    st = np.sqrt(sq_dists(xs, xt).min(axis=1))
    tt = sq_dists(xt, xt)
    np.fill_diagonal(tt, np.inf)
    tt = np.sqrt(tt.min(axis=1))
    return float(np.median(st) / np.median(tt))


def bot_l22_cost(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    fs = polar_bispectrum_features(source, ANGLES)
    ft = polar_bispectrum_features(target, ANGLES)
    return sq_dists(fs, ft)


def solve(mode: str, source: np.ndarray, target: np.ndarray, log: dict) -> np.ndarray:
    nearest = lambda x: polar_bispectrum_features(x, ANGLES)  # noqa: E731
    bot_nearest = lambda: cost_matrix(source, target, nearest, "cityblock", f"bispectral-nearest-{ANGLES}")  # noqa: E731
    bot_bilinear = lambda: cost_matrix(source, target, bilinear_polar_bispectrum, "cityblock", f"bispectral-bilinear-{ANGLES}")  # noqa: E731
    raw = lambda: cost_matrix(source, target, raw_features, "cityblock", "raw")  # noqa: E731
    if mode == "bilinear":
        return entropic_plan(bot_bilinear(), 0.01)
    if mode in ("adaptive", "combined"):
        stat = shift_statistic(source, target)
        shifted = stat > SHIFT_THRESHOLD
        log.update(shift_statistic=stat, invariant_geometry=bool(shifted))
        if not shifted:
            return entropic_plan(raw(), 0.01)
        return entropic_plan(bot_bilinear() if mode == "combined" else bot_nearest(), 0.01)
    if mode == "raw_geometry":
        return entropic_plan(raw(), 0.01)
    if mode == "lower_reg":
        return entropic_plan(bot_nearest(), 0.003)
    if mode == "bot_l22":
        return entropic_plan(bot_l22_cost(source, target), 0.01)
    raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("instances_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()
    logs = {}
    for instance_id, source, target, dataset in iterate_instances(args.instances_dir):
        entry = {"dataset": dataset}
        save_coupling(args.output_dir, instance_id, solve(args.mode, source, target, entry))
        logs[instance_id] = entry
        print(f"[variant:{args.mode}] {instance_id} {entry}", flush=True)
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(json.dumps(logs, indent=2) + "\n")


if __name__ == "__main__":
    main()
