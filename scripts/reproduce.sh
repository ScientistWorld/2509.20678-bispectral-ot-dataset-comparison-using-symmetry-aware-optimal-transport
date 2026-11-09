#!/bin/bash
# Reproduce the reference end to end (runs inside the compute container, CPU).
#
#   1. visible train instances : paper's method (scripts/solve.sh = Bispectral OT, L1)
#      and the evaluator-owned raw-pixel OT baseline -> scoring/scores_train.json
#   2. hidden test instances   : same solver -> scoring/scores_test.json
#   3. scoring/scores.json mirrors the train scores; validate.py checks the workspace.
#
# Assumes data/downloads is populated (scripts/download.sh). ~45 min on 1 CPU.

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p outputs/reference
bash scripts/evaluate_train.sh --baselines-out outputs/reference/baselines_train.json
cp scoring/scores_train.json scoring/scores.json
bash scripts/evaluate_test.sh --baselines-out outputs/reference/baselines_test.json
python validate.py
