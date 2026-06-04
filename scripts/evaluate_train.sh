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
  mapfile -t MANIFESTS < <(find outputs -mindepth 2 -maxdepth 2 -name manifest.json | sort)
  if [ "${#MANIFESTS[@]}" -eq 0 ]; then
    echo "No output manifests found under outputs/*/manifest.json" >&2
    exit 1
  fi
  DUPLICATE_POLICY="keep-last"
elif [ "$#" -eq 1 ]; then
  OUT="scoring/scores_train.json"
  MANIFESTS=("$1")
  DUPLICATE_POLICY="error"
else
  OUT="${@: -1}"
  MANIFESTS=("${@:1:$#-1}")
  DUPLICATE_POLICY="error"
fi

python -m eval.train.evaluate \
  --manifest "${MANIFESTS[@]}" \
  --out "$OUT" \
  --confusion-out scoring/confusion_matrices_train.json \
  --artifact-index scoring/eval_train_artifacts.json \
  --duplicate-policy "$DUPLICATE_POLICY"
