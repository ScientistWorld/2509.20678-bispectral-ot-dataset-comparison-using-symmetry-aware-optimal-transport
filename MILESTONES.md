# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-01 23:56] - core_claim_plus
- Existing MNIST and KMNIST 100-examples-per-class runs execute the paper's actual Bispectral OT algorithm and show BOT improves rotated matching over raw OT.
- Added Fashion-MNIST support to expand Table 1 coverage while preserving the same split, rotation, bispectrum, OT, and class-preservation protocol.

### [2026-06-02 00:04] - core_claim_plus
- Job `48c562eb-931` completed successfully and reproduced the same BOT-over-raw-OT rotated matching pattern on Fashion-MNIST.
- Added EMNIST letters support to cover the remaining Table 1 dataset at scaled sample count.

## Stop Justification
- Not stopping; submitting an EMNIST-inclusive benchmark job.
