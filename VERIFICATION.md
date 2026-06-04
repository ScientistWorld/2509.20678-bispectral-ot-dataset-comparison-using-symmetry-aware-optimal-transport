# Verification Report

## Initial Assessment

The workspace entered final verification at `near_complete`. The implementation and saved artifacts reproduce the paper's central result across MNIST, KMNIST, Fashion-MNIST, and EMNIST letters: rotation-aware dataset alignment substantially improves class-preserving transport over raw-image transport in the rotated condition, while the unrotated controls remain measured as constraints.

The claimed milestone is justified. The remaining gap to `full` is scale: the environment uses 100 examples per class for the packaged reproduction instead of the full Table 1 dataset halves, while keeping the same task, preprocessing, method interface, and scoring protocol.

## Issues Found

- **Minor:** The public `scoring/scores.json` was stale after the evaluator gained nonzero-weight mass diagnostics. It covered all eight experiments, but omitted `same_class_mass_mean`, `class_mass_margin_mean`, and `target_effective_support_ratio`, which were already present in the train/test scores.
- **Cosmetic:** `scoring/TARGETS.md` named the paper method acronym in target descriptions. This was unnecessary for a method-safe guide and could leak more solution framing than needed to from-scratch scientists.

## Fixes Applied

- Regenerated `scoring/scores.json` from existing saved coupling artifacts with a CPU-only scoring job. The refreshed file now includes every nonzero-weight metric for all eight experiments and all eight method entries per experiment.
- Reworded `scoring/TARGETS.md` to describe the experiment outputs, metrics, diagnostics, and constraints without naming the paper method family.
- Verified all URLs in `scripts/download.sh` with `curl -I -L --max-time 15`; each resolved to HTTP 200.
- Ran `python validate.py --compare`; all scoring and structure checks passed.

## Final Judgment

The workspace is ready to serve as a research gym. Improve-mode scientists can reuse the implemented method, artifacts, and scores. From-scratch scientists can use the problem definition, data loaders, evaluation scripts, constraints, and direction documents without depending on `method/`.

The scoring rubric measures the main benefit and the relevant constraints: class preservation, macro accuracy, same-class transported mass, class-mass margin, anti-collapse support, and the prohibition on label-aware costs. No scored metric depends on wall-clock speed or deviation from published values.

- Milestone: near_complete
- Ready for gym use: yes
- Confidence: high
- Key limitation: packaged reproduction uses a reduced 100-example-per-class scale rather than the full Table 1 dataset halves.
