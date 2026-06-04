# Evaluation Targets

## Primary: mnist_rotated_matching

Tests whether transported mass preserves digit labels when matching a randomly rotated MNIST source set to a disjoint unrotated target set.

- **Metric:** class_preservation_accuracy (higher is better), plus mass-distribution diagnostics and the label-cost constraint recorded in `scoring/reference.json`.
- **Expected output:** one manifest containing coupling artifacts for each scored method entry.

## Constraint: mnist_unrotated_baseline

Tests whether the method remains a meaningful dataset alignment procedure when both MNIST source and target sets are unrotated.

- **Metric:** class_preservation_accuracy (higher is better), plus the same diagnostics and constraints as the rotated condition.
- **Expected output:** produced from the same MNIST data split as the rotated condition.

## Additional: kmnist_rotated_matching

Repeats the rotated matching test on KMNIST, a second handwriting dataset from the paper.

- **Metric:** class_preservation_accuracy (higher is better), plus mass-distribution diagnostics and the label-cost constraint.
- **Expected output:** one manifest containing coupling artifacts for each scored method entry.

## Additional Constraint: kmnist_unrotated_baseline

Repeats the unrotated control on KMNIST.

- **Metric:** class_preservation_accuracy (higher is better), plus the same diagnostics and constraints as the rotated condition.
- **Expected output:** produced from the same KMNIST data split as the rotated condition.

## Additional: fmnist_rotated_matching

Repeats the rotated-source benchmark on Fashion-MNIST, the paper's clothing-image benchmark.

- **Metric:** class_preservation_accuracy (higher is better), plus mass-distribution diagnostics and the label-cost constraint.
- **Expected output:** one manifest containing coupling artifacts for each scored method entry.

## Additional Constraint: fmnist_unrotated_baseline

Repeats the unrotated control on Fashion-MNIST.

- **Metric:** class_preservation_accuracy (higher is better), plus the same diagnostics and constraints as the rotated condition.
- **Expected output:** produced from the same Fashion-MNIST data split as the rotated condition.

## Additional: emnist_rotated_matching

Repeats the rotated-source benchmark on the EMNIST letters split, the paper's 26-class handwriting benchmark.

- **Metric:** class_preservation_accuracy (higher is better), plus mass-distribution diagnostics and the label-cost constraint.
- **Expected output:** one manifest containing coupling artifacts for each scored method entry.

## Additional Constraint: emnist_unrotated_baseline

Repeats the unrotated control on EMNIST letters.

- **Metric:** class_preservation_accuracy (higher is better), plus the same diagnostics and constraints as the rotated condition.
- **Expected output:** produced from the same EMNIST data split as the rotated condition.
