#!/bin/bash
# Hardening shortcut 'nan' behind the standard solver interface.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -m redteam.hardening.shortcuts nan "$1" "$2"
