# Train/Test Evaluation Design Notes

The current design was introduced in iteration 11 (see `HARDENING.md`). It replaces the earlier scoring of a stratified half of the rows of method-written couplings, which REDTEAM.md showed could be gamed.

## Interface

- `scripts/solve.sh INSTANCES_DIR OUTPUT_DIR` is the only way a method is scored. Instances (`source`, `target`, `dataset`) carry no labels and no rotated/unrotated flag. Rows are shuffled and ids are opaque. The solver writes one `coupling` per instance.
- `eval/train/evaluate.py` builds the visible instances from the public training splits with a public seed (`--seed`, `--replicates` for fresh development draws, `--key-out` for their answer key).
- `eval/test/evaluate.py` builds the hidden instances from `eval/test/pool/*.npz` (t10k images, labels, angles and order; SHA-256-pinned; regenerable with `eval/test/build_pool.py` and its secret seed). The t10k files are not downloaded into `data/downloads`.
- The two evaluators share no code and use identical instance sizes (100 images per class per side), preprocessing (normalise, then rotate) and validity rules.

## Validity and failure

A coupling must be finite, non-negative, of shape (n, m) and with positive mass, and must have uniform marginals within L1 0.01 after normalising its total mass. Any missing or invalid output, or any exception while reading one, fails the run: floor scores are written after the solver exits and the evaluator returns 1.

## Scores

`scores_{train,test}.json` hold one entry, `method`, per experiment, with `class_preservation_accuracy`, `same_class_mass_mean` and `class_mass_margin_mean`. The raw-pixel OT baseline is computed by the evaluators themselves (`--baselines-out`) and never appears as a scored key.
