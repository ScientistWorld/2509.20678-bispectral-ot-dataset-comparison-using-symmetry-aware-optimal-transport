#!/bin/bash
# Hardening shortcut 'nn_assignment' behind the standard solver interface.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -m redteam.hardening.shortcuts nn_assignment "$1" "$2"
