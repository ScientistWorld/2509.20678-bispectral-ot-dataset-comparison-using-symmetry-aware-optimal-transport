#!/bin/bash
# Hardening shortcut 'forged_extras' behind the standard solver interface.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -m redteam.hardening.shortcuts forged_extras "$1" "$2"
