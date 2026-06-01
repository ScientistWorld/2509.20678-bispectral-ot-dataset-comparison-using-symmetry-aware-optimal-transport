# Overview

- **Paper ID:** 2509.20678
- **Title:** Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport
- **Domain:** Optimal Transport / Dataset Comparison / Geometric Machine Learning
- **TL;DR:** The paper evaluates whether dataset-level transport plans can preserve semantic correspondences when one dataset is transformed by visual symmetries such as rotation.

## Short Summary

The paper studies dataset comparison under nuisance transformations, where ordinary feature distances may match examples by pose or orientation instead of semantic content. Its main benchmark aligns two disjoint halves of image datasets after applying random rotations to one side and measures whether transported mass lands on the correct class. The reported results show large class-preservation gains on MNIST-family datasets in the rotated setting, while also reporting an unrotated-to-unrotated baseline to check that the comparison remains meaningful when no nuisance transformation is present.

## Key Results

- On rotated-to-unrotated matching, the best reported Table 1 accuracies are 0.8405 on MNIST, 0.7815 on KMNIST, 0.7662 on Fashion-MNIST, and 0.5983 on EMNIST letters.
- The naive raw-pixel OT(L1) baseline in the same rotated setting is much lower: 0.3297, 0.2420, 0.3003, and 0.1969 respectively.
- In the unrotated baseline condition, raw-pixel OT(L1) remains strong, so the main claim is specifically about robustness to symmetry-induced nuisance variation rather than simply improving all matching conditions.
