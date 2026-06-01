# Experiment Overview

## mnist_rotated_matching

This is the central benchmark: source MNIST examples are randomly rotated and matched to a disjoint unrotated target set. The score measures whether transported mass preserves digit labels despite nuisance rotation.

## mnist_unrotated_baseline

This control matches two disjoint unrotated MNIST subsets. It checks that a method remains a meaningful dataset alignment tool when there is no rotation shift.

## kmnist_rotated_matching

This repeats the rotated-source benchmark on KMNIST, a second handwriting dataset from the paper.

## kmnist_unrotated_baseline

This repeats the unrotated control on KMNIST.
