#!/bin/bash
# Run the paper's baseline (raw-pixel OT, L1 ground cost) as a solver through
# the same interface and score it on the visible train instances.
# (The evaluators also compute this comparator themselves with --baselines-out.)
#
#   bash scripts/baseline.sh

set -euo pipefail

cd "$(dirname "$0")/.."

bash scripts/evaluate_train.sh --solver baseline/solve.sh \
  --out scoring/baseline_scores_train.json --status-out scoring/baseline_train_status.json \
  --confusion-out scoring/baseline_confusion_train.json "$@"
