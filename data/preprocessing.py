"""Problem-level preprocessing shared by methods and baselines."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


DATASET_STATS = {
    "mnist": (0.1306604787, 0.3081078053),
    "kmnist": (0.1917621507, 0.3483428300),
    "fmnist": (0.2860405969, 0.3530242445),
}


def normalize_images(images: np.ndarray, dataset: str) -> np.ndarray:
    mean, std = DATASET_STATS[dataset]
    return (images.astype(np.float32) / 255.0 - mean) / std


def rotate_images(images: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 360.0, size=len(images)).astype(np.float32)
    rotated = np.empty_like(images, dtype=np.float32)
    for i, (img, angle) in enumerate(zip(images, angles, strict=True)):
        rotated[i] = ndimage.rotate(
            img,
            float(angle),
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
    return rotated, angles


def class_balanced_halves(
    labels: np.ndarray,
    samples_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    left: list[int] = []
    right: list[int] = []
    for cls in sorted(int(c) for c in np.unique(labels)):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        if len(idx) < 2 * samples_per_class:
            raise ValueError(f"class {cls} has {len(idx)} examples, need {2 * samples_per_class}")
        left.extend(idx[:samples_per_class].tolist())
        right.extend(idx[samples_per_class : 2 * samples_per_class].tolist())
    return np.array(left, dtype=np.int64), np.array(right, dtype=np.int64)


def raw_features(images: np.ndarray) -> np.ndarray:
    flat = images.reshape(len(images), -1).astype(np.float32)
    return flat
