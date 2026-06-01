# Evaluation Targets

## Primary: rotated_alignment

Tests whether a dataset alignment method preserves semantic labels when matching a randomly rotated source set to a disjoint unrotated target set from the same dataset.

- **Metric:** class_preservation_accuracy (higher is better)
- **How to run:** Evaluate on MNIST, KMNIST, Fashion-MNIST, and EMNIST letters using the fixed class-balanced split and random-rotation condition.

## Constraint: unrotated_baseline

Tests whether the method remains a meaningful dataset alignment procedure when both source and target sets are unrotated.

- **Metric:** class_preservation_accuracy (higher is better)
- **How to run:** Evaluate on the same four datasets using two disjoint unrotated halves of the training split.

## Baseline: raw_pixel_ot_l1

Tracks the reference raw-feature OT(L1) baseline used in the paper for both the rotated and unrotated settings.

- **Metric:** class_preservation_accuracy (higher is better)
- **How to run:** Evaluate raw-pixel optimal transport with an L1 ground cost under the same data splits and conditions as the primary and constraint targets.
