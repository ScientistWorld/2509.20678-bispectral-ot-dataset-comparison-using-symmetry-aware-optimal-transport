#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -euo pipefail

cd /home/user

L1_METRIC="cityblock"
EXTRA_METRICS="euclidean,sqeuclidean,cosine"

if [ ! -f outputs/bot_mnist/manifest.json ]; then
  python -m method.run_bot --dataset mnist --samples-per-class 100 --metrics "$L1_METRIC" --out-dir outputs/bot_mnist
fi
if [ ! -f outputs/bot_kmnist/manifest.json ]; then
  python -m method.run_bot --dataset kmnist --samples-per-class 100 --metrics "$L1_METRIC" --out-dir outputs/bot_kmnist
fi
if [ ! -f outputs/bot_fmnist/manifest.json ]; then
  python -m method.run_bot --dataset fmnist --samples-per-class 100 --metrics "$L1_METRIC" --out-dir outputs/bot_fmnist
fi
if [ ! -f outputs/bot_emnist/manifest.json ]; then
  python -m method.run_bot --dataset emnist --samples-per-class 100 --metrics "$L1_METRIC" --out-dir outputs/bot_emnist
fi

python -m method.run_bot --dataset mnist --samples-per-class 50 --metrics "$EXTRA_METRICS" --out-dir outputs/bot_mnist_extra_metrics
python -m method.run_bot --dataset kmnist --samples-per-class 50 --metrics "$EXTRA_METRICS" --out-dir outputs/bot_kmnist_extra_metrics
python -m method.run_bot --dataset fmnist --samples-per-class 50 --metrics "$EXTRA_METRICS" --out-dir outputs/bot_fmnist_extra_metrics
python -m method.run_bot --dataset emnist --samples-per-class 50 --metrics "$EXTRA_METRICS" --out-dir outputs/bot_emnist_extra_metrics

bash scripts/evaluate.sh \
  outputs/bot_mnist/manifest.json \
  outputs/bot_kmnist/manifest.json \
  outputs/bot_fmnist/manifest.json \
  outputs/bot_emnist/manifest.json \
  outputs/bot_mnist_extra_metrics/manifest.json \
  outputs/bot_kmnist_extra_metrics/manifest.json \
  outputs/bot_fmnist_extra_metrics/manifest.json \
  outputs/bot_emnist_extra_metrics/manifest.json \
  scoring/scores.json
python validate.py --compare
