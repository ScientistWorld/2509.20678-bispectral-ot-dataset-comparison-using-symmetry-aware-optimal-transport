#!/bin/bash
set -euo pipefail
cd /home/user
# Limit BLAS threads for a predictable CPU allocation; no GPU environment edits.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export AUDIT_RESUME=1
trap 'python -c "from redteam.runner import restore; restore()"' EXIT
for variant in reference weaker_raw_geometry better_control_geometry \
  shortcut_visible_labels shortcut_order_answers shortcut_forged_truth shortcut_tiny_test \
  shortcut_replace_baseline shortcut_remove_rotation shortcut_score_fields shortcut_rebuild_test_answers \
  shortcut_uniform shortcut_zero shortcut_signed_failed_matching shortcut_nan shortcut_empty \
  reference_seed_0 better_lower_reg better_bilinear reference_seed_1 reference_seed_2 better_combined
do
  echo "Running $variant"
  if ! bash "redteam/$variant/run.sh"; then
    echo "Audit case $variant failed; continuing to preserve other evidence."
  fi
  if [ -f redteam/reference/scores_test.json ]; then
    python redteam/rank.py
  fi
done
