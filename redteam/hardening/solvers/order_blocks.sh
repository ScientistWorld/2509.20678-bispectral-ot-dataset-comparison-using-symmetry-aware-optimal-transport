#!/bin/bash
# Hardening shortcut 'order_blocks' behind the standard solver interface.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -m redteam.hardening.shortcuts order_blocks "$1" "$2"
