"""Small IDX readers for MNIST-family datasets."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np


def read_idx_images(path: str | Path) -> np.ndarray:
    path = Path(path)
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an IDX image file; magic={magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols)


def read_idx_labels(path: str | Path) -> np.ndarray:
    path = Path(path)
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an IDX label file; magic={magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n)


def load_training_split(root: str | Path, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    raw = Path(root) / dataset / "raw"
    return (
        read_idx_images(raw / "train-images-idx3-ubyte.gz"),
        read_idx_labels(raw / "train-labels-idx1-ubyte.gz"),
    )
