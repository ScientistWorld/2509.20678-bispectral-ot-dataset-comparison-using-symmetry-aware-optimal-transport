# Evaluation

This evaluation describes the task and its measurement without prescribing a solution method.

## Task Instances

Each instance is a pair of image sets from one dataset (MNIST digits, Kuzushiji-MNIST characters, Fashion-MNIST clothing, or EMNIST letters):

- a **source** set and a disjoint **target** set, each with exactly 100 images per class (1,000 x 1,000 images for the 10-class datasets, 2,600 x 2,600 for the 26 EMNIST letter classes);
- images are 28 x 28, normalised per dataset to mean 0 / standard deviation 1;
- in the **rotated** condition every source image is rotated by an independent angle drawn uniformly from 0-360 degrees (bilinear interpolation about the image centre, out-of-frame pixels filled with 0 after normalisation); in the **unrotated** condition neither set is transformed. The target set is never rotated.

Instances are presented **without labels and without saying which condition they come from**; rows are in random order and instance file names are opaque. The rotated and unrotated instances of a dataset use disjoint images.

## Solver Interface

Every method is scored through one entry point, on the visible and hidden instances alike:

```
bash scripts/solve.sh INSTANCES_DIR OUTPUT_DIR
```

- `INSTANCES_DIR/<id>.npz` contains `source` (float32, n x 28 x 28), `target` (float32, m x 28 x 28) and `dataset` (string).
- For every instance the solver writes `OUTPUT_DIR/<id>.npz` with a single array `coupling` of shape (n, m).
- The solver runs offline on CPU inside the gym container.

A coupling is **valid** if it is real-valued, finite, non-negative, has positive total mass, and, after dividing by its total mass, has row sums 1/n and column sums 1/m up to an L1 error of 0.01 on each side. In other words, it is a transport plan between the uniform distributions on the two sets. If **any** instance's output is missing or invalid, the whole run is scored as failed: every metric of every experiment is set to its floor value (accuracy 0, mass 0, margin -1) and the evaluator exits with status 1. Nothing the solver writes besides `coupling` is read.

## Metrics

All metrics are recomputed by the evaluator from the coupling. Each source image's row is normalised to sum to one, and its mass is summed over the target images of each class.

- **class_preservation_accuracy** (primary, coefficient 1.0, higher is better): fraction of source images whose largest class mass is on their own class. This is the quantity reported in the paper's Table 1.
- **same_class_mass_mean** (coefficient 0.4, higher is better): mean mass a source image sends to targets of its own class, in [0, 1].
- **class_mass_margin_mean** (coefficient 0.2, higher is better): mean of own-class mass minus the largest other-class mass, in [-1, 1].

Each experiment (dataset x condition) averages these over all source images of its instances.

## Experiments and Weights

| Experiment | Weight |
| --- | ---: |
| `{mnist,kmnist,fmnist,emnist}_rotated_matching`: rotated source vs unrotated target | 0.18 each |
| `{mnist,kmnist,fmnist,emnist}_unrotated_baseline`: unrotated vs unrotated control | 0.07 each |

The rotated experiments carry the headline claim. The unrotated controls check that robustness to rotations is not bought by destroying ordinary matching. Because the solver is not told the condition, one method must handle both.

## Visible and Hidden Instances

- `bash scripts/evaluate_train.sh` draws visible instances from the public **training** splits with a public seed (one instance per experiment by default) and writes `scoring/scores_train.json`. `--seed` and `--replicates` draw fresh development instances. `--key-out` writes their answer key; `--keep-workdir` keeps the instance files. `scripts/evaluate.sh` writes the same scores to `scoring/scores.json`.
- `bash scripts/evaluate_test.sh` (no arguments) scores the method on the **hidden** test: two independent instances per experiment drawn from the held-out t10k splits. It writes `scoring/scores_test.json`. These images and their labels are held only by the evaluator.

Both use identical instance sizes and preprocessing, so an honest method scores alike on both. The reference method takes about 20 minutes on one CPU for the hidden test.

## Baselines and Reference

The evaluator itself computes the comparison baseline, optimal transport between raw pixel vectors with an L1 ground cost (entropic, regularisation 0.01), on the same instances. `scoring/reference.json` lists, per experiment, the reference method's hidden-test scores (`method`) and this baseline (`raw_ot_l1`). It also keeps the paper's Table 1 numbers under `_paper_results`.

On the hidden test the baseline reaches 0.17-0.30 accuracy in the rotated condition and 0.69-0.88 in the unrotated one.

The paper's numbers come from full-size halves of the training split (about 3,000 images per class per side). They are not on the gym's scale and are not targets.
