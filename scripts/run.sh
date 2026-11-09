#!/bin/bash
# Reproduce the reference end to end (paper's method through the gym's solver
# interface on the visible train instances and the hidden test, plus the
# evaluator-owned raw-pixel OT baseline). ~45 min on 1 CPU.
#
# The hardening audit (variants, shortcuts, paired-noise ranks) that produced
# HARDENING.md is redteam/hardening/run_audit.sh.

set -euo pipefail

cd /home/user

for path in \
  data/downloads/mnist/raw/train-images-idx3-ubyte.gz \
  data/downloads/kmnist/raw/train-images-idx3-ubyte.gz \
  data/downloads/fmnist/raw/train-images-idx3-ubyte.gz \
  data/downloads/emnist/raw/train-images-idx3-ubyte.gz
do
  test -s "$path"
done

bash scripts/reproduce.sh
