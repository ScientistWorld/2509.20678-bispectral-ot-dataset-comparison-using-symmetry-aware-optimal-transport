#!/bin/bash
# Visible train-slice evaluator.
#
# Mirrors scripts/evaluate.sh's manifest/output argument shape while writing
# scoring/scores_train.json. It reads saved method artifacts and does not import
# from method/.

set -euo pipefail

cd /home/user

mkdir -p scoring

if [ "$#" -eq 0 ]; then
  OUT="scoring/scores_train.json"
  MANIFESTS=("outputs/bot/manifest.json")
elif [ "$#" -eq 1 ]; then
  OUT="scoring/scores_train.json"
  MANIFESTS=("$1")
else
  OUT="${@: -1}"
  MANIFESTS=("${@:1:$#-1}")
fi

python -m eval.train.evaluate \
  --manifest "${MANIFESTS[@]}" \
  --out "$OUT" \
  --confusion-out scoring/confusion_matrices_train.json \
  --artifact-index scoring/eval_train_artifacts.json
