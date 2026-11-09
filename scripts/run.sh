#!/bin/bash
# Fresh final smoke reproduction from tracked source code.
# This reruns the scaled BOT protocol and then evaluates the saved couplings.

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
