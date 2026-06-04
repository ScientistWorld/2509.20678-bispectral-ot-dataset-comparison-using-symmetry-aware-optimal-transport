# Overview

- **Paper ID:** 2509.20678
- **Title:** Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport
- **Domain:** Optimal Transport / Dataset Comparison / Geometric Machine Learning
- **TL;DR:** The paper evaluates whether dataset-level transport plans can preserve semantic correspondences when one dataset is transformed by visual symmetries such as rotation.

## Short Summary

The paper studies dataset comparison under nuisance transformations, where ordinary feature distances may match examples by pose or orientation instead of semantic content. Its main benchmark aligns two disjoint halves of image datasets after applying random rotations to one side and measures whether transported mass lands on the correct class. The reported results show large class-preservation gains on MNIST-family datasets in the rotated setting, while also reporting an unrotated-to-unrotated baseline to check that the comparison remains meaningful when no nuisance transformation is present.

## Key Results

- On rotated-to-unrotated matching, the best reported Table 1 accuracies are 0.8405 on MNIST, 0.7815 on KMNIST, 0.7662 on Fashion-MNIST, and 0.5983 on EMNIST letters.
- The naive raw-pixel OT(L1) baseline in the same rotated setting is much lower: 0.3297, 0.2420, 0.3003, and 0.1969 respectively.
- In the unrotated baseline condition, raw-pixel OT(L1) remains strong, so the main claim is specifically about robustness to symmetry-induced nuisance variation rather than simply improving all matching conditions.

---

# Problem Definition

This benchmark asks whether two finite labeled image datasets can be aligned so that corresponding examples share semantic class labels even when one dataset has been transformed by nuisance rotations. A solution outputs a transport or matching relation between a source set and a target set. Success is measured by how much transported mass from each source class lands on the same target class.

## Research Question

Can a dataset comparison method produce semantic correspondences between transformed and untransformed image datasets without being misled by rotations that change pixel geometry but should not change class identity?

## Why It Matters

Dataset alignment is used in domain adaptation, transfer learning, distribution comparison, and dataset diagnostics. In those settings, a useful correspondence should reflect stable semantic content rather than accidental acquisition conditions such as camera angle, object orientation, or other nuisance transformations.

## Success Criteria

A successful solution should achieve high class-preservation accuracy when matching randomly rotated images to unrotated images from the same dataset. It should also be evaluated on an unrotated-to-unrotated condition so that improved robustness to rotation is not obtained by destroying ordinary within-domain matching quality.

---

# Paper's Method

## Key Contribution

The paper introduces Bispectral Optimal Transport, a symmetry-aware variant of discrete optimal transport that computes the OT cost matrix after representing each image with rotation-invariant bispectral features.

## Approach

Each image is converted from a Cartesian pixel grid into a discretized polar representation. For every radial slice, the method computes a one-dimensional discrete Fourier transform over angular bins, then forms bispectrum entries from triples of Fourier coefficients. These per-radius bispectra are concatenated into a global representation that is invariant to discrete rotations of the angular coordinate.

The method then solves an entropically regularized optimal transport problem between two empirical datasets using pairwise distances between these representations as the ground cost. The resulting coupling is converted into class predictions by aggregating transported mass over target labels and choosing the label receiving the most mass for each source example.

## Main Claims

- **Primary claim:** Transport plans computed in the symmetry-aware feature space preserve semantic class labels much better than raw-feature OT when one dataset is randomly rotated and the other is unrotated.
- **Constraint claim:** The method should remain a meaningful dataset comparison procedure, so it is also evaluated in an unrotated-to-unrotated setting rather than only under the transformed condition.
- **Secondary evidence:** Visualizations and per-class matching statistics suggest that the feature space clusters examples by semantic class while reducing sensitivity to rotation.

---

# Evaluation

This evaluation describes the task and metrics without prescribing a particular solution method.

## Metrics

The main metric is **class-preservation accuracy**. Given a transport coupling from source examples to target examples, aggregate each source row's transported mass by target class and assign the source example to the class with the largest total received mass. Accuracy is the fraction of source examples whose assigned target class equals the source label. Higher is better.

The reference comparison also records the same metric in an unrotated-to-unrotated baseline condition. This checks whether performance under rotations is not gained by making the alignment uninformative in the ordinary no-rotation setting.

## Baselines and Targets

The primary baseline is raw-pixel OT using an L1 ground cost. Table 1 reports this baseline and the paper's method on MNIST, KMNIST, Fashion-MNIST, and EMNIST letters. The rotated-to-unrotated results are the headline targets; the unrotated-to-unrotated results are constraint targets for preserving ordinary matching behavior.

## Evaluation Protocol

Use the torchvision training split for each dataset. Split each class into two disjoint halves with seed 0. For the shifted condition, apply uniformly sampled rotations from 0 to 360 degrees to one half and leave the other half unrotated, then compute a transport plan from the rotated set to the unrotated set. For the baseline condition, match the two unrotated halves.

Report class-preservation accuracy for each dataset and condition. The paper's Table 1 uses MNIST, KMNIST, Fashion-MNIST, and EMNIST letters; all images are normalized to mean 0 and standard deviation 1 before evaluation.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 16m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 7 total (7 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-01 23:56] - core_claim_plus
- Existing MNIST and KMNIST 100-examples-per-class runs execute the paper's actual Bispectral OT algorithm and show BOT improves rotated matching over raw OT.
- Added Fashion-MNIST support to expand Table 1 coverage while preserving the same split, rotation, bispectrum, OT, and class-preservation protocol.

### [2026-06-02 00:04] - core_claim_plus
- Job `48c562eb-931` completed successfully and reproduced the same BOT-over-raw-OT rotated matching pattern on Fashion-MNIST.
- Added EMNIST letters support to cover the remaining Table 1 dataset at scaled sample count.

### [2026-06-02 00:10] - core_claim_plus
- Job `23fc5045-020` completed successfully and reproduced the BOT-over-raw-OT rotated matching pattern on EMNIST letters at 50 examples per class.
- Scheduling EMNIST at 100 examples per class to improve the scaled reproduction quality.

### [2026-06-02 00:23] - core_claim_plus
- Job `b1483a5b-018` completed successfully and improved the scaled EMNIST reproduction to 100 examples per class.
- The core Table 1 pattern is reproduced at scaled size on MNIST, KMNIST, Fashion-MNIST, and EMNIST letters: BOT improves rotated class preservation over raw-pixel OT without using labels in the cost.

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `core_claim` | **Status**: error
- **Working time**: 14m | **GPU**: 0.0h
- **Jobs**: 7 total (7 completed, 0 failed)

### Iteration 3: gpt-5.5
- **Milestone**: `near_complete` | **Status**: done
- **Working time**: 15m | **GPU**: 0.0h
- **Jobs**: 10 total (10 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `near_complete` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 10 total (10 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `near_complete` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 10 total (10 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `near_complete` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 12 total (12 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `near_complete` | **Status**: done
- **Working time**: 2m | **GPU**: 0.0h
- **Jobs**: 12 total (12 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `near_complete` | **Status**: done
- **Working time**: 8m | **GPU**: 0.0h
- **Jobs**: 13 total (13 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>


---

# Reproduction Milestones

**Current: near_complete**
