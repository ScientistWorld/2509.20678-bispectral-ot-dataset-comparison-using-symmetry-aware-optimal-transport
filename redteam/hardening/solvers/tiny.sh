#!/bin/bash
# Hardening shortcut 'tiny' behind the standard solver interface.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -m redteam.hardening.shortcuts tiny "$1" "$2"
