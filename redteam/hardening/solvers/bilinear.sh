#!/bin/bash
# Hardening variant 'bilinear' behind the standard solver interface.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -m redteam.hardening.variants bilinear "$1" "$2" ${HARDENING_LOG:+--log "$HARDENING_LOG"}
