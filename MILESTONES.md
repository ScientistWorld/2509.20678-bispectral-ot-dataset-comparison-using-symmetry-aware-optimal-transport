# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-04 07:03] - core_claim_plus
- Existing CPU jobs executed the paper's BOT pipeline end-to-end on MNIST, KMNIST, Fashion-MNIST, and EMNIST letters at scaled sample counts.
- Reproduced the central effect: bispectral OT preserves labels much better than raw pixel OT when the source set is randomly rotated.
- Added the paper's full raw/BOT distance-metric matrix to `scoring/reference.json`; the next submitted job expands reproduced scores beyond L1.
