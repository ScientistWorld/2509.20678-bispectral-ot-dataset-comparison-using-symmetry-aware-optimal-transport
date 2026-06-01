#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -euo pipefail

cd /home/user

DATASET="${1:-mnist}"
SAMPLES_PER_CLASS="${2:-20}"
OUT_DIR="${3:-outputs/bot}"

python -m method.run_bot \
  --dataset "$DATASET" \
  --samples-per-class "$SAMPLES_PER_CLASS" \
  --out-dir "$OUT_DIR"

bash scripts/evaluate.sh "$OUT_DIR/manifest.json" scoring/scores.json
