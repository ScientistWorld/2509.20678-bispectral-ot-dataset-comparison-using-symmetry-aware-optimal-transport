# Evaluation

This evaluation describes the task and metrics without prescribing a particular solution method.

## Metrics

The main metric is **class-preservation accuracy**. Given a transport coupling from source examples to target examples, aggregate each source row's transported mass by target class and assign the source example to the class with the largest total received mass. Accuracy is the fraction of source examples whose assigned target class equals the source label. Higher is better.

The reference comparison also records the same metric in an unrotated-to-unrotated baseline condition. This checks whether performance under rotations is not gained by making the alignment uninformative in the ordinary no-rotation setting.

## Baselines and Targets

The primary baseline is raw-pixel OT using an L1 ground cost. Table 1 reports this baseline and the paper's method on MNIST, KMNIST, Fashion-MNIST, and EMNIST letters. The rotated-to-unrotated results are the headline targets; the unrotated-to-unrotated results are constraint targets for preserving ordinary matching behavior.

## Evaluation Protocol

Use the torchvision training split for each dataset. Split each class into two disjoint halves with seed 0. For the shifted condition, apply uniformly sampled rotations from 0 to 360 degrees to one half and leave the other half unrotated, then compute a transport plan from the rotated set to the unrotated set. For the baseline condition, match the two unrotated halves.

Report class-preservation accuracy for each dataset and condition. The paper's Table 1 uses MNIST, KMNIST, Fashion-MNIST, and EMNIST letters; all images are normalized to mean 0 and standard deviation 1 before evaluation.
