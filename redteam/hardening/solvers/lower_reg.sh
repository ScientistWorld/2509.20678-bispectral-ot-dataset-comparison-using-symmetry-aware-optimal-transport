#!/bin/bash
# Hardening variant 'lower_reg' behind the standard solver interface.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -m redteam.hardening.variants lower_reg "$1" "$2" ${HARDENING_LOG:+--log "$HARDENING_LOG"}
