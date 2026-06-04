#!/bin/bash
# Run the ResearchGym train/test scoring split against all discovered final
# method artifacts. The split scripts discover manifests under outputs/.

set -euo pipefail

cd /home/user

bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
