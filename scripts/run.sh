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

python -m method.run_bot --dataset mnist --samples-per-class 20 --out-dir outputs/bot_mnist
python -m method.run_bot --dataset kmnist --samples-per-class 20 --out-dir outputs/bot_kmnist
bash scripts/evaluate.sh \
  outputs/bot_mnist/manifest.json \
  outputs/bot_kmnist/manifest.json \
  scoring/scores.json
python validate.py --compare
