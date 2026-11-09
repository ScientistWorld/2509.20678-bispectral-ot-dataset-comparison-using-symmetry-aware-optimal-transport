#!/bin/bash
# Hardening variant 'adaptive' behind the standard solver interface.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -m redteam.hardening.variants adaptive "$1" "$2" ${HARDENING_LOG:+--log "$HARDENING_LOG"}
