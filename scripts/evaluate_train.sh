#!/bin/bash
# Visible train evaluation: builds public matching instances from the training
# splits, runs the solver (scripts/solve.sh) on them through the same
# interface as the hidden test, validates every coupling and recomputes all
# scores into scoring/scores_train.json.
#
#   bash scripts/evaluate_train.sh                     # default public instances
#   bash scripts/evaluate_train.sh --seed 7 --replicates 2 --out tmp/dev.json \
#        --key-out tmp/dev_key.json --keep-workdir     # fresh development draw
#
# See python -m eval.train.evaluate --help for every flag.
# Exit status 1 means the run FAILED validation (floor scores written).

set -euo pipefail

cd "$(dirname "$0")/.."

python -m eval.train.evaluate "$@"
