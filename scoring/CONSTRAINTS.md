# Scientist Constraints

## Data

Use only the standard torchvision training splits for MNIST, KMNIST, Fashion-MNIST, and EMNIST letters. Do not add external labeled images, extra test data, synthetic class examples beyond the specified rotation transform, or pretrained classifiers trained on related image datasets. The benchmark is about alignment under controlled nuisance transformations, so extra labeled data would change the problem rather than improve the method.

Split each dataset by class into two disjoint halves using seed 0. The target labels may be used only for scoring the transport output, not for constructing a supervised target-label cost or training a classifier.

## Evaluation Setting

The rotated condition must match one randomly rotated half to the other unrotated half. Rotations should be sampled uniformly over 0 to 360 degrees as in the paper's benchmark. The unrotated baseline condition must match the two unrotated halves.

All submitted scores must report class-preservation accuracy for all four datasets and both conditions. Do not replace the task with image classification, retrieval against the same examples, or matching examples that are not from the fixed disjoint halves.

## Resource Scope

The method may change the representation, distance, regularization, solver details, or alignment procedure, but it must remain a dataset alignment method that outputs a coupling or equivalent source-to-target mass assignment. It must not rely on a large pretrained vision model or an external supervised recognizer to assign labels directly.

## Compliance Checklist

- [ ] Uses only torchvision MNIST, KMNIST, Fashion-MNIST, and EMNIST letters training splits
- [ ] Splits each class into two disjoint halves with seed 0
- [ ] Does not use target labels to build the alignment method
- [ ] Reports rotated-to-unrotated and unrotated-to-unrotated class-preservation accuracy
- [ ] Produces a dataset alignment or transport-like source-to-target assignment, not standalone classification
- [ ] Does not add external labeled data or pretrained class-recognition models
