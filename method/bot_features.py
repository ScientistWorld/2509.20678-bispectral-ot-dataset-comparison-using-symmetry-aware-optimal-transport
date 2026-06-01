"""Bispectral feature construction used by Bispectral OT."""

from __future__ import annotations

import math

import numpy as np


def polar_bispectrum_features(images: np.ndarray, num_angles: int = 40) -> np.ndarray:
    if images.ndim != 3:
        raise ValueError("expected images with shape (N, H, W)")
    features = [_single_polar_bispectrum(img, num_angles) for img in images]
    return np.stack(features).astype(np.float32)


def _single_polar_bispectrum(img: np.ndarray, num_angles: int) -> np.ndarray:
    h, w = img.shape
    radial_bins = min(h, w) // 2
    polar = np.zeros((radial_bins, num_angles), dtype=np.float32)
    for r in range(radial_bins):
        for k in range(num_angles):
            theta = 2.0 * math.pi * k / num_angles
            x = int(w / 2.0 + r * math.cos(theta))
            y = int(h / 2.0 + r * math.sin(theta))
            if 0 <= x < w and 0 <= y < h:
                polar[r, k] = (r**1) * img[y, x]

    fft = np.fft.fft(polar, axis=1)
    idx = (np.arange(num_angles)[:, None] + np.arange(num_angles)[None, :]) % num_angles
    bispec = fft[:, :, None] * fft[:, None, :] * np.conj(fft[:, idx])
    norm = np.linalg.norm(bispec.ravel())
    if norm > 0:
        bispec = bispec / norm
    return np.concatenate([bispec.real.ravel(), bispec.imag.ravel()]).astype(np.float32)
