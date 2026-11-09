# Evaluation Targets

All targets are scored by `scripts/evaluate_test.sh` (hidden instances) and mirrored on visible instances by `scripts/evaluate_train.sh`. Each experiment reports `class_preservation_accuracy` (primary), `same_class_mass_mean` and `class_mass_margin_mean`, all higher-is-better and recomputed from the solver's coupling.

**Expected output for every target:** for each instance, `OUTPUT_DIR/<id>.npz` with `coupling`: a non-negative, finite (n, m) transport plan whose marginals are uniform within L1 0.01 after normalising total mass. A missing or invalid plan fails every target.

## Primary: mnist_rotated_matching

Class preservation when a randomly rotated MNIST source set is matched to a disjoint unrotated target set (1,000 x 1,000 images per instance). Reference (hidden test): accuracy 0.7215; raw-pixel OT baseline 0.2995.

## Primary: kmnist_rotated_matching

Same on Kuzushiji-MNIST. Reference 0.5095; baseline 0.2155.

## Primary: fmnist_rotated_matching

Same on Fashion-MNIST. Reference 0.7070; baseline 0.2785.

## Primary: emnist_rotated_matching

Same on EMNIST letters (26 classes, 2,600 x 2,600 images per instance). Reference 0.4279; baseline 0.1675.

## Constraint: mnist_unrotated_baseline

Two disjoint unrotated MNIST sets. Rotation robustness must not destroy ordinary matching, and the solver is not told which condition it faces. Reference 0.7085; baseline 0.8765.

## Constraint: kmnist_unrotated_baseline

Reference 0.5335; baseline 0.8300.

## Constraint: fmnist_unrotated_baseline

Reference 0.7160; baseline 0.7765.

## Constraint: emnist_unrotated_baseline

Reference 0.4431; baseline 0.6896.

## Validity constraint (all experiments)

The output must be a transport plan between the two uniform empirical measures (see above). Violations score the run as failed, with every metric at its floor.
