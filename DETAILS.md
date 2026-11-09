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
- The gym runs the same protocol at 100 images per class per side, on hidden held-out instances. There the reference reaches 0.59 mean rotated accuracy against 0.24 for raw-pixel OT, so the paper's numbers serve as context rather than targets (see `briefing/evaluation.md`).

---

# Problem Definition

This benchmark asks whether two finite image datasets can be aligned so that corresponding examples share semantic class labels even when one dataset has been transformed by nuisance rotations. A solution outputs a transport plan between a source set and a target set, with mass spread uniformly over each set's images. Success is measured by how much transported mass from each source image lands on target images of the same class.

## Research Question

Can a dataset comparison method produce semantic correspondences between transformed and untransformed image datasets without being misled by rotations that change pixel geometry but should not change class identity?

## Why It Matters

Dataset alignment is used in domain adaptation, transfer learning, distribution comparison, and dataset diagnostics. In those settings, a useful correspondence should reflect stable semantic content rather than accidental acquisition conditions such as camera angle, object orientation, or other nuisance transformations.

## Success Criteria

A successful solution achieves high class-preservation accuracy when matching randomly rotated images to unrotated images from the same dataset. The same solution must also match two unrotated sets well, so that robustness to rotation is not obtained by destroying ordinary within-domain matching quality. The solver is never told which of the two situations it faces, and it never sees labels.

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

## In This Gym

`scripts/solve.sh` runs `method/solve.py`. For each instance it builds the bispectral features (`method/bot_features.py`: R = 14 radii, K = 40 angles, nearest-pixel polar sampling at the image centre, radius weighting, per-image unit norm). It then computes the L1 ground cost between features, normalised by its maximum, and solves entropic OT with POT's Greenkhorn at regularisation 0.01 (up to 10^6 iterations, tolerance 1e-9). The coupling is the solver's output.

On the hidden test this reference reaches 0.43-0.72 class-preservation accuracy in the rotated condition, against 0.17-0.30 for raw-pixel OT. In the unrotated control it trails raw-pixel OT (0.44-0.72 vs 0.69-0.88), the same direction as the paper's Table 1.

---

# Evaluation

This evaluation describes the task and its measurement without prescribing a solution method.

## Task Instances

Each instance is a pair of image sets from one dataset (MNIST digits, Kuzushiji-MNIST characters, Fashion-MNIST clothing, or EMNIST letters):

- a **source** set and a disjoint **target** set, each with exactly 100 images per class (1,000 x 1,000 images for the 10-class datasets, 2,600 x 2,600 for the 26 EMNIST letter classes);
- images are 28 x 28, normalised per dataset to mean 0 / standard deviation 1;
- in the **rotated** condition every source image is rotated by an independent angle drawn uniformly from 0-360 degrees (bilinear interpolation about the image centre, out-of-frame pixels filled with 0 after normalisation); in the **unrotated** condition neither set is transformed. The target set is never rotated.

Instances are presented **without labels and without saying which condition they come from**; rows are in random order and instance file names are opaque. The rotated and unrotated instances of a dataset use disjoint images.

## Solver Interface

Every method is scored through one entry point, on the visible and hidden instances alike:

```
bash scripts/solve.sh INSTANCES_DIR OUTPUT_DIR
```

- `INSTANCES_DIR/<id>.npz` contains `source` (float32, n x 28 x 28), `target` (float32, m x 28 x 28) and `dataset` (string).
- For every instance the solver writes `OUTPUT_DIR/<id>.npz` with a single array `coupling` of shape (n, m).
- The solver runs offline on CPU inside the gym container.

A coupling is **valid** if it is real-valued, finite, non-negative, has positive total mass, and, after dividing by its total mass, has row sums 1/n and column sums 1/m up to an L1 error of 0.01 on each side. In other words, it is a transport plan between the uniform distributions on the two sets. If **any** instance's output is missing or invalid, the whole run is scored as failed: every metric of every experiment is set to its floor value (accuracy 0, mass 0, margin -1) and the evaluator exits with status 1. Nothing the solver writes besides `coupling` is read.

## Metrics

All metrics are recomputed by the evaluator from the coupling. Each source image's row is normalised to sum to one, and its mass is summed over the target images of each class.

- **class_preservation_accuracy** (primary, coefficient 1.0, higher is better): fraction of source images whose largest class mass is on their own class. This is the quantity reported in the paper's Table 1.
- **same_class_mass_mean** (coefficient 0.4, higher is better): mean mass a source image sends to targets of its own class, in [0, 1].
- **class_mass_margin_mean** (coefficient 0.2, higher is better): mean of own-class mass minus the largest other-class mass, in [-1, 1].

Each experiment (dataset x condition) averages these over all source images of its instances.

## Experiments and Weights

| Experiment | Weight |
| --- | ---: |
| `{mnist,kmnist,fmnist,emnist}_rotated_matching`: rotated source vs unrotated target | 0.18 each |
| `{mnist,kmnist,fmnist,emnist}_unrotated_baseline`: unrotated vs unrotated control | 0.07 each |

The rotated experiments carry the headline claim. The unrotated controls check that robustness to rotations is not bought by destroying ordinary matching. Because the solver is not told the condition, one method must handle both.

## Visible and Hidden Instances

- `bash scripts/evaluate_train.sh` draws visible instances from the public **training** splits with a public seed (one instance per experiment by default) and writes `scoring/scores_train.json`. `--seed` and `--replicates` draw fresh development instances. `--key-out` writes their answer key; `--keep-workdir` keeps the instance files. `scripts/evaluate.sh` writes the same scores to `scoring/scores.json`.
- `bash scripts/evaluate_test.sh` (no arguments) scores the method on the **hidden** test: two independent instances per experiment drawn from the held-out t10k splits. It writes `scoring/scores_test.json`. These images and their labels are held only by the evaluator.

Both use identical instance sizes and preprocessing, so an honest method scores alike on both. The reference method takes about 20 minutes on one CPU for the hidden test.

## Baselines and Reference

The evaluator itself computes the comparison baseline, optimal transport between raw pixel vectors with an L1 ground cost (entropic, regularisation 0.01), on the same instances. `scoring/reference.json` lists, per experiment, the reference method's hidden-test scores (`method`) and this baseline (`raw_ot_l1`). It also keeps the paper's Table 1 numbers under `_paper_results`.

On the hidden test the baseline reaches 0.17-0.30 accuracy in the rotated condition and 0.69-0.88 in the unrotated one.

The paper's numbers come from full-size halves of the training split (about 3,000 images per class per side). They are not on the gym's scale and are not targets.

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

### Iteration 9: gpt-5.5 (verifier)
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

### [2026-06-04 08:31] - near_complete
- Updated `scripts/run.sh` to perform a fresh scaled reproduction from tracked method code via `scripts/reproduce.sh`, rather than only rescoring existing output artifacts.
- Prepared a CPU test submission to regenerate BOT couplings and validate the scoring contract end-to-end.

### [2026-06-04 08:35] - near_complete
- Final verification passed with `python validate.py --compare`.
- Confirmed method/problem separation, train/test evaluator wiring, download URL accessibility, and scoring rubric sanity.
- Fixed the public evaluator's no-argument mode so it discovers available output manifests instead of scoring only one default manifest.

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `near_complete` | **Status**: done
- **Working time**: 12m | **GPU**: 0.0h
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

### [2026-06-04 08:31] - near_complete
- Updated `scripts/run.sh` to perform a fresh scaled reproduction from tracked method code via `scripts/reproduce.sh`, rather than only rescoring existing output artifacts.
- Prepared a CPU test submission to regenerate BOT couplings and validate the scoring contract end-to-end.

### [2026-06-04 08:35] - near_complete
- Final verification passed with `python validate.py --compare`.
- Confirmed method/problem separation, train/test evaluator wiring, download URL accessibility, and scoring rubric sanity.
- Fixed the public evaluator's no-argument mode so it discovers available output manifests instead of scoring only one default manifest.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `near_complete` | **Status**: done
- **Working time**: 27m | **GPU**: 0.0h
- **Jobs**: 14 total (14 completed, 0 failed)

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

### [2026-06-04 08:31] - near_complete
- Updated `scripts/run.sh` to perform a fresh scaled reproduction from tracked method code via `scripts/reproduce.sh`, rather than only rescoring existing output artifacts.
- Prepared a CPU test submission to regenerate BOT couplings and validate the scoring contract end-to-end.

### [2026-06-04 08:35] - near_complete
- Final verification passed with `python validate.py --compare`.
- Confirmed method/problem separation, train/test evaluator wiring, download URL accessibility, and scoring rubric sanity.
- Fixed the public evaluator's no-argument mode so it discovers available output manifests instead of scoring only one default manifest.

### [2026-11-09 01:30] - near_complete (hardening, iteration 11)
- Rebuilt the evaluation so the evaluator owns the answers. The hidden test pool (t10k images and labels, secret selection) lives only in `eval/test/pool/`. Instances are shuffled, unlabeled and condition-blind, and every method is scored through `scripts/solve.sh`.
- Couplings must be valid transport plans with uniform marginals, otherwise the run fails with floor scores. The raw-OT baseline is computed by `eval/`.
- Job `cbda84be-9df`: the reference reproduces the paper's direction on all four datasets (rotated 0.592 vs 0.240 for raw OT). Better variants beat it by 15-38 paired SEs, weaker ones fall below, and every REDTEAM.md shortcut now fails or scores chance. See `HARDENING.md`.
- The milestone stays `near_complete`: the method is unchanged and full-dataset scale is still out of reach.

</details>


---

# Reproduction Milestones

**Current: near_complete**
