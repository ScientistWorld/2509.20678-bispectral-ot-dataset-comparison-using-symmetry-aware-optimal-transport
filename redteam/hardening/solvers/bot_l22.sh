#!/bin/bash
# Hardening variant 'bot_l22' behind the standard solver interface.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -m redteam.hardening.variants bot_l22 "$1" "$2" ${HARDENING_LOG:+--log "$HARDENING_LOG"}
