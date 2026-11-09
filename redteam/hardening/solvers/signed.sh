#!/bin/bash
# Hardening shortcut 'signed' behind the standard solver interface.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -m redteam.hardening.shortcuts signed "$1" "$2"
