# Scientist Constraints

Use the fixed per-class source and target split for each run, and report class-preservation accuracy from saved coupling artifacts. The target labels may be used only by `eval/` for scoring, never to construct a label-aware transport cost.

The rotated condition must match one randomly rotated half to the other unrotated half. The unrotated baseline condition must match the two unrotated halves. A submission must remain a dataset alignment method that outputs a coupling or equivalent source-to-target mass assignment, not a standalone classifier.

The metric `label_cost_used` must remain 0. This guards against trivial improvements that inject target labels into the cost matrix rather than solving the paper's unsupervised alignment problem.
