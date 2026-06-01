#!/bin/bash
# Run a baseline method from the paper.
#
# Baselines are simple reference methods the paper compares against. 
# Implement this only if applicable, i.e. the paper has baselines.
# You do not need to implement all baselines to begin — 
# just enough to validate that the evaluation pipeline works 
# and provide a reference point. You can decide to expand more baselines later.
# Each baseline must be actually run and scored through evaluate.sh
# so the numbers are verified, not self-reported.
# You can decide what this baseline script should do yourself.
#
# Usage:
#   bash scripts/baseline.sh <name>

set -euo pipefail

cd /home/user

DATASET="${1:-mnist}"
SAMPLES_PER_CLASS="${2:-20}"
OUT_DIR="${3:-outputs/baseline}"

python -m baseline.run_raw_ot \
  --dataset "$DATASET" \
  --samples-per-class "$SAMPLES_PER_CLASS" \
  --out-dir "$OUT_DIR"

bash scripts/evaluate.sh "$OUT_DIR/manifest.json" scoring/baseline_scores.json
