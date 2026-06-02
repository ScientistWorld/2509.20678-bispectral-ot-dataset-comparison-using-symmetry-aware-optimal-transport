#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0.597 GB
# Estimated disk usage after extraction: 0.622 GB compressed
#
# This portable script downloads to workspace-local paths so a future user
# with just this repository can repopulate the required data.

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p data/downloads/mnist/raw data/downloads/kmnist/raw data/downloads/fmnist/raw data/downloads/emnist/raw

fetch() {
  local url="$1"
  local out="$2"
  if [ ! -s "$out" ]; then
    curl -L "$url" -o "$out"
  fi
}

fetch https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz data/downloads/mnist/raw/train-images-idx3-ubyte.gz
fetch https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz data/downloads/mnist/raw/train-labels-idx1-ubyte.gz
fetch https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz data/downloads/mnist/raw/t10k-images-idx3-ubyte.gz
fetch https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz data/downloads/mnist/raw/t10k-labels-idx1-ubyte.gz

fetch https://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz data/downloads/kmnist/raw/train-images-idx3-ubyte.gz
fetch https://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz data/downloads/kmnist/raw/train-labels-idx1-ubyte.gz
fetch https://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz data/downloads/kmnist/raw/t10k-images-idx3-ubyte.gz
fetch https://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz data/downloads/kmnist/raw/t10k-labels-idx1-ubyte.gz

fetch https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz data/downloads/fmnist/raw/train-images-idx3-ubyte.gz
fetch https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz data/downloads/fmnist/raw/train-labels-idx1-ubyte.gz
fetch https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz data/downloads/fmnist/raw/t10k-images-idx3-ubyte.gz
fetch https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz data/downloads/fmnist/raw/t10k-labels-idx1-ubyte.gz

fetch https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip data/downloads/emnist/gzip.zip

python - <<'PY'
from pathlib import Path
import zipfile

root = Path("data/downloads/emnist")
raw = root / "raw"
mapping = {
    "gzip/emnist-letters-train-images-idx3-ubyte.gz": raw / "train-images-idx3-ubyte.gz",
    "gzip/emnist-letters-train-labels-idx1-ubyte.gz": raw / "train-labels-idx1-ubyte.gz",
    "gzip/emnist-letters-test-images-idx3-ubyte.gz": raw / "t10k-images-idx3-ubyte.gz",
    "gzip/emnist-letters-test-labels-idx1-ubyte.gz": raw / "t10k-labels-idx1-ubyte.gz",
}
with zipfile.ZipFile(root / "gzip.zip") as zf:
    for member, out in mapping.items():
        if out.exists() and out.stat().st_size > 0:
            continue
        out.write_bytes(zf.read(member))
PY
