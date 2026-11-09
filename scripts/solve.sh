#!/bin/bash
# Solver entry point: the ONE interface through which every method is scored,
# on the visible train instances and on the hidden test instances alike.
#
#   bash scripts/solve.sh INSTANCES_DIR OUTPUT_DIR
#
# INSTANCES_DIR holds <id>.npz files with arrays
#     source  float32 (n, 28, 28)  normalised images (possibly rotated)
#     target  float32 (m, 28, 28)  normalised images
#     dataset str                  mnist | kmnist | fmnist | emnist
# No labels and no rotated/unrotated flag are provided; row order is random.
# For every instance write OUTPUT_DIR/<id>.npz with one array
#     coupling (n, m)  non-negative transport plan with uniform marginals
# (see briefing/evaluation.md for the exact validity rules).
#
# It runs offline (no internet) on CPU inside the gym container. Replace the
# command below to submit a different method.

set -euo pipefail
cd "$(dirname "$0")/.."

python -m method.solve "$1" "$2"
