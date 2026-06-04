#!/bin/bash
# Held-out test-slice evaluator.
#
# Mirrors scripts/evaluate.sh's manifest/output argument shape while writing
# scoring/scores_test.json. By default it reads the train-produced artifact
# index and scoring/scores_train.json, then scores the same experiment/method
# entries through eval/test/.

set -euo pipefail

cd /home/user

mkdir -p scoring

if [ "$#" -eq 0 ]; then
  OUT="scoring/scores_test.json"
  MANIFESTS=()
elif [ "$#" -eq 1 ]; then
  OUT="scoring/scores_test.json"
  MANIFESTS=("$1")
else
  OUT="${@: -1}"
  MANIFESTS=("${@:1:$#-1}")
fi

if [ "${#MANIFESTS[@]}" -eq 0 ]; then
  python -m eval.test.evaluate \
    --out "$OUT" \
    --confusion-out scoring/confusion_matrices_test.json \
    --train-scores scoring/scores_train.json \
    --artifact-index scoring/eval_train_artifacts.json
else
  python -m eval.test.evaluate \
    --manifest "${MANIFESTS[@]}" \
    --out "$OUT" \
    --confusion-out scoring/confusion_matrices_test.json \
    --train-scores scoring/scores_train.json \
    --artifact-index scoring/eval_train_artifacts.json
fi
