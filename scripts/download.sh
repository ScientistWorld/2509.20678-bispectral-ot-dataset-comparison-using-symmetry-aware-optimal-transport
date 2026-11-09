#!/bin/bash
# Download all public data needed by this environment (idempotent).
#
# Run on a node with internet access BEFORE building the container or
# submitting compute jobs. Skips files that already exist.
#
# Only the TRAINING splits are fetched into data/downloads/. The held-out test
# images come from the public t10k splits, but the gym ships its hidden test
# selection (images + labels) inside eval/test/pool/ (committed, regenerable
# with eval/test/build_pool.py); the t10k files are deliberately not placed
# where a solver can read them.
#
# Estimated total download size: ~0.62 GB (the 0.56 GB EMNIST archive is
#   deleted after the letters training files are extracted)
# Estimated disk usage after extraction: ~0.09 GB

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p data/downloads/mnist/raw data/downloads/kmnist/raw data/downloads/fmnist/raw data/downloads/emnist/raw

fetch() {
  local url="$1"
  local out="$2"
  if [ ! -s "$out" ]; then
    curl -fL --retry 3 "$url" -o "$out.part"
    mv "$out.part" "$out"
  fi
}

fetch https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz data/downloads/mnist/raw/train-images-idx3-ubyte.gz
fetch https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz data/downloads/mnist/raw/train-labels-idx1-ubyte.gz

fetch https://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz data/downloads/kmnist/raw/train-images-idx3-ubyte.gz
fetch https://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz data/downloads/kmnist/raw/train-labels-idx1-ubyte.gz

fetch https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz data/downloads/fmnist/raw/train-images-idx3-ubyte.gz
fetch https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz data/downloads/fmnist/raw/train-labels-idx1-ubyte.gz

EMNIST_RAW=data/downloads/emnist/raw
if [ ! -s "$EMNIST_RAW/train-images-idx3-ubyte.gz" ] || [ ! -s "$EMNIST_RAW/train-labels-idx1-ubyte.gz" ]; then
  fetch https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip data/downloads/emnist/gzip.zip
  python3 - <<'PY'
from pathlib import Path
import zipfile

root = Path("data/downloads/emnist")
raw = root / "raw"
mapping = {
    "gzip/emnist-letters-train-images-idx3-ubyte.gz": raw / "train-images-idx3-ubyte.gz",
    "gzip/emnist-letters-train-labels-idx1-ubyte.gz": raw / "train-labels-idx1-ubyte.gz",
}
with zipfile.ZipFile(root / "gzip.zip") as zf:
    for member, out in mapping.items():
        out.write_bytes(zf.read(member))
PY
  rm -f data/downloads/emnist/gzip.zip
fi
