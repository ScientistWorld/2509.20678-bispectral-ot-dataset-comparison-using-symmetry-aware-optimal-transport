# Research Direction

## Research Problem

This benchmark is about symmetry-robust dataset comparison: how to align two empirical image distributions so that the correspondence reflects semantic content when one distribution may have been transformed by nuisance rotations.

## Core Contribution

The paper contributes a symmetry-aware way to define the geometry used by optimal transport. A scientist should work on the representation, ground cost, invariance, or transport formulation used for symmetry-robust dataset alignment, while keeping the problem framing: unsupervised transport between two finite datasets with uniform weights.

## Approach Scope

In-bounds work stays within dataset alignment by transport plans. Examples:

- alter the invariant representation (sampling, interpolation, normalisation, other invariants);
- use different rotation-aware distances or costs;
- change the OT regularisation or solver, as long as the output is a valid transport plan with uniform marginals;
- infer from the unlabeled data whether, and how strongly, the two sets differ by a transformation, and adapt the geometry accordingly. The solver is not told which condition an instance comes from, so one method must serve both.

Methods may be learned or non-learned, but any learning must be unsupervised (no class labels).

The open problems are the rotated-condition accuracy, which the reference leaves well below 1 on every dataset, and the gap between invariant matching and raw-pixel matching on unrotated data.

## Out of Bounds

- Replacing alignment with supervised image classification, or using class labels anywhere (costs, representations, tuning, post-processing, lookups in labelled files).
- Reading or modifying `eval/`, `scoring/`, or anything the evaluator writes.
- Using external pretrained vision classifiers or embeddings whose main benefit is prior supervised recognition on similar datasets.
- Outputting anything other than a transport plan with uniform marginals, such as nearest-neighbour assignments or partial plans.
