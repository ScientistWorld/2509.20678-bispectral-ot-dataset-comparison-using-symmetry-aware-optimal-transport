#!/bin/bash
# Run the paper's method (Bispectral OT, L1 ground cost) through the gym's
# solver interface on the visible train instances and score it.
#
#   bash scripts/method.sh

set -euo pipefail

cd "$(dirname "$0")/.."

bash scripts/evaluate_train.sh --solver scripts/solve.sh "$@"
