# Experiment Overview

Every experiment scores one transport plan per instance, produced by the solver behind `scripts/solve.sh` from unlabeled images. The solver is never told whether an instance is rotated. Scores are recomputed by the evaluator. An invalid plan anywhere fails the whole run. On the hidden test each experiment has two independent instances drawn from held-out images.

## mnist_rotated_matching (weight 0.18)

The central benchmark. Source MNIST digits are randomly rotated (uniform 0-360 degrees) and matched to a disjoint unrotated target set. It establishes whether transported mass preserves digit identity despite nuisance rotation.

## mnist_unrotated_baseline (weight 0.07)

The control. Two disjoint unrotated MNIST sets are matched. It establishes that a method stays a good ordinary alignment tool when no rotation is present, so rotation robustness is not bought by discarding information that matters without rotations.

## kmnist_rotated_matching (weight 0.18)

The rotated benchmark on Kuzushiji-MNIST cursive characters: a second handwriting domain with different stroke statistics.

## kmnist_unrotated_baseline (weight 0.07)

The unrotated control on Kuzushiji-MNIST.

## fmnist_rotated_matching (weight 0.18)

The rotated benchmark on Fashion-MNIST, where rotation acts like an object-pose nuisance on clothing images.

## fmnist_unrotated_baseline (weight 0.07)

The unrotated control on Fashion-MNIST.

## emnist_rotated_matching (weight 0.18)

The rotated benchmark on EMNIST letters. It raises the number of classes from 10 to 26, and several letters are near-rotations of each other, so it probes the hardest end of the claim.

## emnist_unrotated_baseline (weight 0.07)

The unrotated control on EMNIST letters.

## Metrics (all experiments)

- `class_preservation_accuracy` (coefficient 1.0): the paper's quantity, the share of source images whose transported mass is largest on their own class.
- `same_class_mass_mean` (0.4) and `class_mass_margin_mean` (0.2): how much of each image's mass reaches its own class, and how decisively. Both distinguish confident from barely-correct correspondences.
