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
# Estimated total download size: 0.061 GB
# Estimated disk usage after extraction: 0.061 GB compressed
#
# This portable script downloads to workspace-local paths so a future user
# with just this repository can repopulate the required data.

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p data/downloads/mnist/raw data/downloads/kmnist/raw data/downloads/fmnist/raw

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
