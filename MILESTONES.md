# Reproduction Milestones

**Current: core_claim_plus**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

### [2026-06-01 23:34] - none
- Read the paper and upstream code release.
- Implemented the Bispectral OT computation and artifact-based evaluator.
- Downloaded MNIST/KMNIST data and prepared a CPU scheduled smoke job.

### [2026-06-01 23:40] - core_claim
- Scheduled CPU smoke job completed successfully with the paper's Bispectral OT algorithm.
- On scaled MNIST rotated matching with 20 examples per class, raw OT reached 0.305 class-preservation accuracy and Bispectral OT reached 0.605.
- The unrotated control also ran end-to-end: raw OT reached 0.815 and Bispectral OT reached 0.635, matching the paper's expected trade-off direction.

### [2026-06-01 23:43] - core_claim_plus
- Expanded the same scaled protocol to KMNIST in job `1d3643cc-26f`.
- On scaled KMNIST rotated matching with 20 examples per class, raw OT reached 0.210 class-preservation accuracy and Bispectral OT reached 0.435.
- The merged evaluator now scores all four configured MNIST/KMNIST experiments in one `scoring/scores.json`.

### [2026-06-01 23:46] - core_claim_plus
- Increased MNIST and KMNIST to 50 examples per class in job `55a1ecdd-4af`.
- Rotated MNIST improved to 0.674 class-preservation accuracy for Bispectral OT versus 0.310 for raw OT.
- Rotated KMNIST improved to 0.492 class-preservation accuracy for Bispectral OT versus 0.218 for raw OT.

### [2026-06-01 23:53] - core_claim_plus
- Increased MNIST and KMNIST to 100 examples per class in job `2e3b5935-44d`.
- Rotated MNIST reached 0.694 class-preservation accuracy for Bispectral OT versus 0.297 for raw OT.
- Rotated KMNIST reached 0.564 class-preservation accuracy for Bispectral OT versus 0.203 for raw OT.
- Final environment is packaged with method, baseline, evaluator, downloads, scoring references, and reproducible scripts.

## Stop Justification
- Completed at milestone `core_claim_plus`.
