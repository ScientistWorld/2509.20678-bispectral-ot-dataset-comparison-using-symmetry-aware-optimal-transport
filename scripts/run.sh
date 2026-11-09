#!/bin/bash
# Temporary audit job entry point; restored to the initial revision afterwards.
set -euo pipefail
cd /home/user
bash redteam/run_all.sh
