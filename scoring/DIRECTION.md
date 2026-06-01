# Research Direction

## Research Problem

This benchmark is about symmetry-robust dataset comparison. The central question is how to align two empirical image distributions so that the correspondence reflects semantic content when one distribution has been transformed by nuisance rotations.

## Core Contribution

The paper contributes a symmetry-aware way to define the geometry used by optimal transport. A scientist should work on improving the representation, cost, invariance, or transport formulation used for symmetry-robust dataset alignment, while preserving the same problem framing: unsupervised matching between two finite datasets.

## Approach Scope

In-bounds work should remain within dataset alignment and transport-like comparison. A scientist may alter the invariant representation, use different rotation-aware distances, change the OT regularization or solver, introduce principled normalization, or improve how transported mass is converted into label-level correspondence for evaluation.

Methods may be learned or non-learned, but they must solve the alignment problem without directly training a label classifier for the benchmark labels. The scientific question is whether better geometry yields better correspondences under nuisance transformations.

## Out of Bounds

- Replacing alignment with supervised image classification.
- Using target labels to construct the cost matrix, train the representation, tune per-class matching, or post-process assignments.
- Using external pretrained vision classifiers or embeddings whose main benefit is prior supervised recognition on similar datasets.
- Changing the task to same-image retrieval, paired-example matching, or a different transformation family without also reporting the required rotation benchmark.
- Reporting only the rotated condition while omitting the unrotated baseline condition.
