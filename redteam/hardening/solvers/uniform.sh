#!/bin/bash
# Hardening shortcut 'uniform' behind the standard solver interface.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -m redteam.hardening.shortcuts uniform "$1" "$2"
