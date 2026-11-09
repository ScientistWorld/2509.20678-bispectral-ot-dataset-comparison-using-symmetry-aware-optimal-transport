#!/bin/bash
set -euo pipefail
cd /home/user
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export AUDIT_RESUME=0
trap 'python -c "from redteam.runner import restore; restore()"' EXIT
bash redteam/shortcut_empty/run.sh
bash redteam/shortcut_rebuild_test_answers/run.sh
bash redteam/better_combined/run.sh
python redteam/rank.py
