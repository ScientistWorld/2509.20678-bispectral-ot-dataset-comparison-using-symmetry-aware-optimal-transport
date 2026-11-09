"""Build the hidden test pool (run once; the result is committed in eval/test/pool/).

The pool holds the only copy of the test images and their labels that the
workspace keeps. Images come from the public *t10k* (held-out) splits of MNIST,
KMNIST, Fashion-MNIST and EMNIST-letters, which are NOT downloaded into
data/downloads by scripts/download.sh. The selection seed below is the only
secret; with it and the public files this script regenerates the pool exactly.

Pure Python (no numpy) so the pool bytes do not depend on library versions.

Layout per dataset (``pool/<dataset>.npz``), for replicate r in {0, 1} and
condition c in {rotated, unrotated}:
    r{r}_{c}_source_images  uint8 (n, 28, 28)   already in presentation order
    r{r}_{c}_source_labels  int64 (n,)
    r{r}_{c}_target_images  uint8 (n, 28, 28)
    r{r}_{c}_target_labels  int64 (n,)
    r{r}_rotated_source_angles float32 (n,)     degrees, uniform on [0, 360)
with n = 100 * num_classes. All 8 blocks of a dataset use disjoint images.

Usage:
    python3 eval/test/build_pool.py --raw-root <dir with <dataset>/t10k-*.gz>
Missing t10k files are fetched from the public URLs into --raw-root.
"""

from __future__ import annotations

import argparse
import array
import gzip
import hashlib
import io
import json
import random
import struct
import urllib.request
import zipfile
from pathlib import Path

SEED = "bispectral-ot-hidden-test::7f3c91e2a4d8"
PER_CLASS = 100
REPLICATES = 2
CONDITIONS = ("rotated", "unrotated")
SIDES = ("source", "target")

URLS = {
    "mnist": (
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
    ),
    "kmnist": (
        "https://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz",
        "https://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz",
    ),
    "fmnist": (
        "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz",
        "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz",
    ),
    # EMNIST letters test files live inside the official gzip.zip archive.
    "emnist": (
        "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip#gzip/emnist-letters-test-images-idx3-ubyte.gz",
        "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip#gzip/emnist-letters-test-labels-idx1-ubyte.gz",
    ),
}


def _fetch(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    base, _, member = url.partition("#")
    if member:
        archive = out.parent / Path(base).name
        if not archive.exists():
            urllib.request.urlretrieve(base, archive)
        with zipfile.ZipFile(archive) as zf:
            out.write_bytes(zf.read(member))
    else:
        urllib.request.urlretrieve(base, out)


def _read_idx(path: Path, magic_expected: int) -> tuple[list[int], bytes]:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic = struct.unpack(">I", raw[:4])[0]
    if magic != magic_expected:
        raise ValueError(f"{path}: magic {magic} != {magic_expected}")
    ndim = magic & 0xFF
    dims = list(struct.unpack(">" + "I" * ndim, raw[4 : 4 + 4 * ndim]))
    return dims, raw[4 + 4 * ndim :]


def _npy(name: str, payload: bytes, descr: str, shape: tuple[int, ...]) -> tuple[str, bytes]:
    header = "{'descr': '%s', 'fortran_order': False, 'shape': %r, }" % (descr, shape)
    total = 10 + len(header) + 1
    header += " " * ((64 - total % 64) % 64) + "\n"
    blob = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin1") + payload
    return name + ".npy", blob


def build_dataset(dataset: str, raw_root: Path) -> tuple[list[tuple[str, bytes]], dict]:
    img_path = raw_root / dataset / "t10k-images-idx3-ubyte.gz"
    lab_path = raw_root / dataset / "t10k-labels-idx1-ubyte.gz"
    for url, path in zip(URLS[dataset], (img_path, lab_path)):
        if not path.exists():
            _fetch(url, path)
    (n_img, rows, cols), pixels = _read_idx(img_path, 2051)
    (n_lab,), labels_raw = _read_idx(lab_path, 2049)
    assert n_img == n_lab and rows == 28 and cols == 28
    labels = list(labels_raw)
    if dataset == "emnist":
        labels = [x - 1 for x in labels]  # letters are encoded 1..26
    classes = sorted(set(labels))
    size = rows * cols
    rng = random.Random(f"{SEED}:{dataset}")
    by_class = {c: [i for i, y in enumerate(labels) if y == c] for c in classes}
    blocks_needed = REPLICATES * len(CONDITIONS) * len(SIDES)
    chosen = {c: rng.sample(by_class[c], blocks_needed * PER_CLASS) for c in classes}

    members: list[tuple[str, bytes]] = []
    block = 0
    for r in range(REPLICATES):
        for cond in CONDITIONS:
            for side in SIDES:
                idx = [i for c in classes for i in chosen[c][block * PER_CLASS : (block + 1) * PER_CLASS]]
                block += 1
                rng.shuffle(idx)
                n = len(idx)
                prefix = f"r{r}_{cond}_{side}"
                img_bytes = b"".join(pixels[i * size : (i + 1) * size] for i in idx)
                members.append(_npy(f"{prefix}_images", img_bytes, "|u1", (n, rows, cols)))
                members.append(_npy(f"{prefix}_labels", array.array("q", [labels[i] for i in idx]).tobytes(), "<i8", (n,)))
                if cond == "rotated" and side == "source":
                    angles = array.array("f", [rng.uniform(0.0, 360.0) for _ in idx])
                    members.append(_npy(f"r{r}_rotated_source_angles", angles.tobytes(), "<f4", (n,)))
    meta = {"classes": len(classes), "per_class_per_block": PER_CLASS, "replicates": REPLICATES}
    return members, meta


def write_npz(path: Path, members: list[tuple[str, bytes]]) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name, blob in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, blob)
    path.write_bytes(buf.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path(__file__).resolve().parent / "pool", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for dataset in ("mnist", "kmnist", "fmnist", "emnist"):
        members, meta = build_dataset(dataset, args.raw_root)
        out = args.out_dir / f"{dataset}.npz"
        write_npz(out, members)
        meta["sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
        manifest[dataset] = meta
        print(dataset, meta)
    (args.out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
