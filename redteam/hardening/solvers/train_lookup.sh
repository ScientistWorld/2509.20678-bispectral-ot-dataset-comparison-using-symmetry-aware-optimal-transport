#!/bin/bash
# Hardening shortcut 'train_lookup' behind the standard solver interface.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -m redteam.hardening.shortcuts train_lookup "$1" "$2"
