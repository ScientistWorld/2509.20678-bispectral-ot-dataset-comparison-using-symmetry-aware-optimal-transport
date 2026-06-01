# Reproduction Milestones

**Current: core_claim**

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

## Stop Justification
- Not stopped. Next job expands the same protocol to KMNIST.
