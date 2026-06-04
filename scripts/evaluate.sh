#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# This script must be SELF-CONTAINED using only code in eval/.
# Do NOT import from method/ — evaluate.sh must work even when the
# paper's method code is hidden from a scientist agent.
#
# Design it so that anyone following the same output format can run
# this script to evaluate their results.
#
# OUTPUT CONTRACT: This script MUST write scoring/scores.json containing
# a JSON dict nested by experiment, e.g.:
#   {"imagenet_classification": {"top1_accuracy": 82.1, "top5_accuracy": 94.8}}
# Experiment and metric names must match the keys in scoring/reference.json.
# This file is read by the system for automated comparison.
#
# Example usage:
#   ./scripts/evaluate.sh    # discover outputs/*/manifest.json
#   ./scripts/evaluate.sh outputs/bot_mnist/manifest.json scoring/scores.json
#   ./scripts/evaluate.sh outputs/bot_mnist/manifest.json outputs/bot_kmnist/manifest.json scoring/scores.json
#
# Called from run.sh after training, or independently.

set -euo pipefail

cd /home/user

mkdir -p scoring

if [ "$#" -eq 0 ]; then
  OUT="scoring/scores.json"
  mapfile -t MANIFESTS < <(find outputs -maxdepth 2 -name manifest.json -print | sort)
  if [ "${#MANIFESTS[@]}" -eq 0 ]; then
    echo "No output manifests found under outputs/*/manifest.json" >&2
    exit 1
  fi
  DUPLICATE_POLICY="keep-last"
elif [ "$#" -eq 1 ]; then
  OUT="scoring/scores.json"
  MANIFESTS=("$1")
  DUPLICATE_POLICY="error"
else
  OUT="${@: -1}"
  MANIFESTS=("${@:1:$#-1}")
  DUPLICATE_POLICY="error"
fi

python -m eval.evaluate_artifacts \
  --manifest "${MANIFESTS[@]}" \
  --duplicate-policy "$DUPLICATE_POLICY" \
  --out "$OUT" \
  --confusion-out scoring/confusion_matrices.json
