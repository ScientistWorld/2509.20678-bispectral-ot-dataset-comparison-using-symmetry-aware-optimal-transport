# Reproduction Milestones

**Current: near_complete**

## Progress Log

### [2026-06-04 07:03] - core_claim_plus
- Existing CPU jobs executed the paper's BOT pipeline end-to-end on MNIST, KMNIST, Fashion-MNIST, and EMNIST letters at scaled sample counts.
- Reproduced the central effect: bispectral OT preserves labels much better than raw pixel OT when the source set is randomly rotated.
- Added the paper's full raw/BOT distance-metric matrix to `scoring/reference.json`; the next submitted job expands reproduced scores beyond L1.

### [2026-06-04 07:18] - secondary_claims
- Job `36fa2488-5ce` completed successfully and filled `scoring/scores.json` with real artifacts for L2, squared L2, and cosine raw/BOT OT variants across MNIST, KMNIST, Fashion-MNIST, and EMNIST.
- The extra distance metrics reproduce the same qualitative claim as Table 2: bispectral features preserve class labels better than raw pixels under random rotations.
- A follow-up CPU job is queued to rerun these non-L1 variants at 100 examples per class, matching the current L1 scale.

### [2026-06-04 07:49] - majority
- Job `c49239cf-dbd` completed successfully and reran the non-L1 variants at 100 examples per class.
- The reproduced score file now covers all 64 raw/BOT distance-metric entries in the reference matrix across MNIST, KMNIST, Fashion-MNIST, and EMNIST.
- Full-dataset scale and appendix confusion matrices remain the only substantive gaps.

### [2026-06-04 07:53] - near_complete
- Job `1b9763e1-c49` completed successfully and generated `scoring/confusion_matrices.json` from the saved coupling artifacts.
- The environment now includes the paper's algorithm, all benchmark datasets, all raw/BOT cost variants, rotated and unrotated controls, and class-level diagnostic matrices.
- The only remaining gap to `full` is running the same protocol on complete dataset halves instead of the budgeted 100 examples per class.

### [2026-06-04 08:18] - near_complete
- Expanded the public artifact evaluator so `scoring/scores.json` can include every diagnostic metric declared in `scoring/reference.json`.
- Prepared a CPU test job that regenerates scores from the existing 100-example coupling artifacts and runs `validate.py --compare` inside the container.
- The milestone remains `near_complete`; this is a scoring-contract verification, not a new full-scale experiment.

### [2026-06-04 08:25] - near_complete
- Job `fafa5075-dce` completed successfully and regenerated public scores from saved coupling artifacts.
- All eight experiments now include every nonzero-weight diagnostic metric in `scoring/scores.json`.
- Final verification passed: download URLs resolve, method/eval/data separation is clean, train/test score files are aligned, and `python validate.py --compare` reports all checks passed.
