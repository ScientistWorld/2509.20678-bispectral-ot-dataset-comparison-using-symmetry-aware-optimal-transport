#!/bin/bash
# Public evaluation entry point: scores the method behind scripts/solve.sh on
# the visible train instances and writes scoring/scores.json (same numbers as
# scoring/scores_train.json). Uses only eval/ code; never imports method/.
#
#   bash scripts/evaluate.sh [extra flags for eval.train.evaluate]

set -euo pipefail

cd "$(dirname "$0")/.."

python -m eval.train.evaluate --out scoring/scores.json --status-out scoring/train_status.json "$@"
