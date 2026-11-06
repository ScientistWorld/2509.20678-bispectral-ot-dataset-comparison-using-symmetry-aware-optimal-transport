# Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2509.20678"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2509.20678-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2509.20678](https://arxiv.org/abs/2509.20678) |
| **Domain** | Optimal Transport / Dataset Comparison / Geometric Machine Learning |
| **TL;DR** | The paper evaluates whether dataset-level transport plans can preserve semantic correspondences when one dataset is transformed by visual symmetries such as rotation. |

## Summary

The paper studies dataset comparison under nuisance transformations, where ordinary feature distances may match examples by pose or orientation instead of semantic content. Its main benchmark aligns two disjoint halves of image datasets after applying random rotations to one side and measures whether transported mass lands on the correct class. The reported results show large class-preservation gains on MNIST-family datasets in the rotated setting, while also reporting an unrotated-to-unrotated baseline to check that the comparison remains meaningful when no nuisance transformation is present.

## Problem

Can a dataset comparison method produce semantic correspondences between transformed and untransformed image datasets without being misled by rotations that change pixel geometry but should not change class identity?

## The paper's method

The paper introduces Bispectral Optimal Transport, a symmetry-aware variant of discrete optimal transport that computes the OT cost matrix after representing each image with rotation-invariant bispectral features.

## Key results of the paper

- On rotated-to-unrotated matching, the best reported Table 1 accuracies are 0.8405 on MNIST, 0.7815 on KMNIST, 0.7662 on Fashion-MNIST, and 0.5983 on EMNIST letters.
- The naive raw-pixel OT(L1) baseline in the same rotated setting is much lower: 0.3297, 0.2420, 0.3003, and 0.1969 respectively.
- In the unrotated baseline condition, raw-pixel OT(L1) remains strong, so the main claim is specifically about robustness to symmetry-induced nuisance variation rather than simply improving all matching conditions.

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.6-terra-xhigh` | +70.1 |
| `gpt-5.5-xhigh` | -62.2 |
| `gpt-5.6-sol-xhigh` | -132.4 |
| `gpt-5.6-luna-xhigh` | -135.3 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install sciworld
sw research 2509.20678 --mode innovate --agent claude-code
sw evaluate 2509.20678
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

