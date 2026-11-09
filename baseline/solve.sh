#!/bin/bash
# Raw-pixel OT baseline behind the standard solver interface.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m baseline.raw_ot "$1" "$2"
