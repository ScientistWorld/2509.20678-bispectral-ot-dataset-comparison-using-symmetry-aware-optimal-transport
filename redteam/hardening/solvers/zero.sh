#!/bin/bash
# Hardening shortcut 'zero' behind the standard solver interface.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -m redteam.hardening.shortcuts zero "$1" "$2"
