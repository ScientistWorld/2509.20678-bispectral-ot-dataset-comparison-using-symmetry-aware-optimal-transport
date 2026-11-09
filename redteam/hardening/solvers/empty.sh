#!/bin/bash
# Hardening shortcut 'empty' behind the standard solver interface.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -m redteam.hardening.shortcuts empty "$1" "$2"
