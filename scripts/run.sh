#!/bin/bash
# Re-score the final 100-example BOT artifacts with the public evaluator.
# This job does not rerun OT; it verifies the saved couplings and scoring
# contract inside the container environment.

set -euo pipefail

cd /home/user

bash scripts/evaluate.sh \
  outputs/bot_mnist/manifest.json \
  outputs/bot_mnist_extra_metrics_100/manifest.json \
  outputs/bot_kmnist/manifest.json \
  outputs/bot_kmnist_extra_metrics_100/manifest.json \
  outputs/bot_fmnist/manifest.json \
  outputs/bot_fmnist_extra_metrics_100/manifest.json \
  outputs/bot_emnist/manifest.json \
  outputs/bot_emnist_extra_metrics_100/manifest.json \
  scoring/scores.json

python validate.py --compare
