#!/bin/bash
# Hidden test: builds the held-out matching instances that only eval/test/
# holds, runs the solver (scripts/solve.sh) on them, validates every coupling
# and recomputes all scores into scoring/scores_test.json.
#
#   bash scripts/evaluate_test.sh            # scores the method behind scripts/solve.sh
#
# Optional flags (see python -m eval.test.evaluate --help): --solver PATH,
# --out PATH, --baselines-out PATH, --details-out PATH.
# Exit status 1 means the run FAILED validation (floor scores written).

set -euo pipefail

cd "$(dirname "$0")/.."

python -m eval.test.evaluate "$@"
