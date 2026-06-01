# Evaluation Targets

## Primary: mnist_rotated_matching

Tests whether transported mass preserves digit labels when matching a randomly rotated MNIST source set to a disjoint unrotated target set.

- **Metric:** class_preservation_accuracy (higher is better)
- **How to run:** `bash scripts/method.sh mnist <samples_per_class> <out_dir>`

## Constraint: mnist_unrotated_baseline

Tests whether the method remains a meaningful dataset alignment procedure when both MNIST source and target sets are unrotated.

- **Metric:** class_preservation_accuracy (higher is better)
- **How to run:** Produced by the same MNIST method run.

## Additional: kmnist_rotated_matching

Repeats the rotated matching test on KMNIST, a second handwriting dataset from the paper.

- **Metric:** class_preservation_accuracy (higher is better)
- **How to run:** `bash scripts/method.sh kmnist <samples_per_class> <out_dir>`

## Additional Constraint: kmnist_unrotated_baseline

Repeats the unrotated control on KMNIST.

- **Metric:** class_preservation_accuracy (higher is better)
- **How to run:** Produced by the same KMNIST method run.
