#!/bin/bash
# Hardening variant 'raw_geometry' behind the standard solver interface.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -m redteam.hardening.variants raw_geometry "$1" "$2" ${HARDENING_LOG:+--log "$HARDENING_LOG"}
