# Reproduction Milestones

**Current: secondary_claims**

## Progress Log

### [2026-06-04 07:03] - core_claim_plus
- Existing CPU jobs executed the paper's BOT pipeline end-to-end on MNIST, KMNIST, Fashion-MNIST, and EMNIST letters at scaled sample counts.
- Reproduced the central effect: bispectral OT preserves labels much better than raw pixel OT when the source set is randomly rotated.
- Added the paper's full raw/BOT distance-metric matrix to `scoring/reference.json`; the next submitted job expands reproduced scores beyond L1.

### [2026-06-04 07:18] - secondary_claims
- Job `36fa2488-5ce` completed successfully and filled `scoring/scores.json` with real artifacts for L2, squared L2, and cosine raw/BOT OT variants across MNIST, KMNIST, Fashion-MNIST, and EMNIST.
- The extra distance metrics reproduce the same qualitative claim as Table 2: bispectral features preserve class labels better than raw pixels under random rotations.
- A follow-up CPU job is queued to rerun these non-L1 variants at 100 examples per class, matching the current L1 scale.
