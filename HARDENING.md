# Hardening report (iteration 11)

**Verdict: yes, better results now score higher.** On the hardened hidden test every genuine improvement ranks above the reference and every weaker variant below it, all by at least 10 paired standard errors. Every shortcut from `REDTEAM.md` now fails the run or lands at chance. The reference scores what it did before: its rotated accuracy is 0.592 vs 0.599 on the old test.

All numbers come from one CPU job (`cbda84be-9df`, exit 0, 91 min), run by `redteam/hardening/run_audit.sh`. Every attempt went through the same entry point the hidden test uses (`python -m eval.test.evaluate --solver ...`). The reference went through `bash scripts/evaluate_test.sh` itself, and a final no-argument rerun reproduced its scores exactly (`results/final_noarg_check.json`). Per-attempt scores, statuses and logs are in `redteam/hardening/results/<name>/`. The ranking and noise analysis is in `redteam/hardening/summary.json` and `analysis.txt`.

## What was wrong and what changed

REDTEAM.md found one root cause behind every major shortcut: **the method wrote the answer key and the test.** The coupling artifacts carried the source/target labels, the sample counts and the class-sorted row order, and the evaluator trusted all of them. The "hidden" test was a public, recomputable half of the rows the method had already produced. The raw-OT comparator artifacts were method-written and counted in the rank, nothing validated the coupling, and the experiment name told the method which condition it faced.

| Fix | Where |
| --- | --- |
| **Hidden test pool.** 2 replicates x {rotated, unrotated} x {source, target} x 100 images/class from the public **t10k** splits, selected with a secret seed. It is stored with labels, angles and row order only in `eval/test/pool/*.npz` (12.6 MB, SHA-256-pinned in the evaluator; regenerable with `eval/test/build_pool.py`). `scripts/download.sh` no longer fetches any t10k file and deletes the EMNIST archive after extracting the training letters; the local copies were removed. | `eval/test/`, `scripts/download.sh` |
| **Evaluator-built instances.** Rows are shuffled, with no labels, no condition flag and opaque ids. The rotated and unrotated instances use disjoint images, so a solver cannot pair a rotated image with its unrotated twin. | `eval/{train,test}/evaluate.py` |
| **One solver interface for train and test.** `bash scripts/solve.sh INSTANCES_DIR OUTPUT_DIR`; only the `coupling` array is read. The method is training-free, so the hidden test runs it on instances only `eval/test/` holds (about 18 min of solver time for the reference). | `scripts/solve.sh`, `method/solve.py` |
| **Validity before scoring.** A coupling must be real, finite, non-negative, of shape (n, m), with positive mass and uniform marginals within L1 0.01 after normalising total mass. Any violation fails the run: floor scores in every experiment (accuracy 0, mass 0, margin −1), rewritten after the solver finishes, and exit status 1. Any exception while reading an output counts as invalid. | both evaluators |
| **Comparator owned by eval/.** Raw-pixel OT (L1) is computed by the evaluator (`--baselines-out`) and is no longer a scored result key. `scores_*.json` holds a single entry, `method`, per experiment. | both evaluators, `scoring/reference.json` |
| **Metrics that can move.** Removed `label_cost_used` (hard-coded 0, enforced nothing), `mean_diagonal_class_accuracy` (identical to accuracy on class-balanced instances), `target_effective_support_ratio` (≈1 for every valid plan, now enforced as a constraint) and the zero-coefficient counts. | `scoring/reference.json` |

Train and test share no code. Train draws from the public training split with a public seed. Test draws from t10k images that appear nowhere else on disk.

## Do better results score higher? (variants before and after)

Rank delta = Σ weight × |coef| × (attempt − reference) / std over the valid attempts of this audit (12 attempts, `summary.json`). "Paired SE" is the std of that delta over 300 bootstraps that resample hidden-test rows with the same indices for attempt and reference. Accuracies are weight-averaged over the four datasets.

**Before** (old test; ranks from REDTEAM.md over its 20-attempt population and 384 coordinates):

| Variant | Rank Δ | Rotated acc | Unrotated acc | Note |
| --- | ---: | ---: | ---: | --- |
| combined (bilinear + raw geometry for the *named* unrotated condition) | +1.650 | 0.5687 | 0.8033 | read the condition from the experiment name |
| control geometry (raw geometry for the *named* unrotated condition) | +1.378 | 0.5478 | 0.8033 | same |
| bilinear polar sampling | +0.391 | 0.5687 | 0.5702 | **inside seed spread 0.603**: not detectable |
| lower regularization 0.003 | −1.210 | 0.5160 | 0.5178 | |
| raw geometry | −3.206 | 0.2222 | 0.8033 | |

**After** (hardened hidden test):

| Variant | Rank Δ | Paired SE | Rotated acc | Unrotated acc | Train rot / unrot |
| --- | ---: | ---: | ---: | ---: | ---: |
| combined: bilinear + **data-driven** shift detection | **+0.566** | 0.015 | 0.6153 | 0.7932 | 0.6124 / 0.7967 |
| adaptive: detect a shift from the unlabeled sets (pixel NN-distance ratio > 1.1); raw-pixel OT if none, BOT otherwise | **+0.374** | 0.007 | 0.5915 | 0.7932 | 0.5870 / 0.7967 |
| bilinear polar sampling at the true rotation centre | **+0.215** | 0.014 | 0.6153 (+0.0238 ± 0.0033) | 0.6112 | 0.6124 / 0.6293 |
| *reference: BOT, L1, reg 0.01* | 0 | — | 0.5915 | 0.6003 | 0.5870 / 0.6197 |
| lower regularization 0.003 (honest failed trial) | −0.147 | 0.014 | 0.5449 | 0.5515 | — |
| BOT with the paper's squared-L2 cost | −0.413 | 0.019 | 0.5201 | 0.5285 | — |
| raw geometry (no bispectrum) | −1.469 | 0.029 | 0.2402 | 0.7932 | 0.2432 / 0.7967 |

- The order is unchanged whether σ is taken over the valid audit population, over every attempt including the failed ones, or over the genuine variants only (`rank_delta_*` in `summary.json`).
- The bilinear fix, invisible to the old test, now beats the reference by 15 paired SEs. Its rotated gain repeats on the independent train draw (+0.025).
- Lower regularization raises same-class mass (0.513 vs 0.448) but lowers accuracy, and correctly ranks below the reference: accuracy dominates the score.
- The detection threshold in adaptive/combined was fixed in advance, not tuned on the test. Its statistic is 0.99–1.00 on unrotated instances and 1.12–1.62 on rotated ones (`results/adaptive/solver_log_test.json`).

## Major shortcuts: fix and score before → after

| Shortcut (REDTEAM.md) | Before | Fix | After (hidden test) |
| --- | ---: | --- | --- |
| Visible labels in the artifact | +20.53 | Instances carry no labels; only `coupling` is read. The closest analog, `train_lookup` (look every image up in the labelled public files), finds 0% of test images because the t10k files are not on disk | −2.947 (chance 0.085); **1.000 on train unrotated**, so train-tuned loses on test |
| Order answers (`index // 100`) | +20.53 | Rows shuffled by the evaluator | `order_blocks` −3.020 (accuracy 0.083) |
| Rebuild test answers from the public split code | +20.53 | Test instances exist only in `eval/test/pool`; nothing public regenerates them | not reproducible; see `train_lookup` |
| Forged truth / extra arrays | +20.53 | Truth comes from the pool; extra arrays are ignored | `forged_extras` −2.947 (chance) |
| Tiny test (fewer rows) | +20.53 | Sizes fixed by the evaluator; wrong shape fails the run | `tiny` **failed** (floor, exit 1) |
| Remove rotation (submit unrotated couplings as rotated) | +11.40 | The evaluator rotates the images and hides the condition; disjoint images per condition | not expressible; the honest equivalent `raw_geometry` scores −1.469 |
| Replace baseline artifacts | +2.45 | Baseline computed by `eval/`, not a scored key | not expressible |
| Signed failed matching (negative mass) | +26.92 | Non-negativity check | `signed` **failed** |
| Score fields / overwrite scoring files | 0 (tie by replay) | Every score is recomputed after the solver exits; the floor is rewritten on failure | `forged_extras` overwrote `scoring/*.json` with perfect values; the evaluator replaced them: −2.947 |
| Condition read from the experiment name | +1.378 | Opaque instance ids; no condition flag | no longer possible; a data-driven equivalent is a legitimate method (adaptive, above) |
| NaN coupling | undefined (NaN accepted) | Finiteness check | `nan` **failed** |
| Zero coupling | −21.50 | Positive-mass check | `zero` **failed** |
| Empty output | rejected | Missing file fails the run | `empty` **failed** |
| Uniform coupling | −14.01 | (valid plan, no information) | `uniform` −2.947 (chance) |
| *New:* nearest-neighbour assignment (not a transport plan) | — | Uniform-marginal check | `nn_assignment` **failed** |

Failed runs score −4.806 (floor in every metric), below every valid attempt including the information-free ones (−2.95), and exit with status 1.

## Shortcuts left open

- **Reading the evaluator's files.** The hidden pool, with its labels, sits in `eval/test/pool/` in the container that runs the test. A solver that searches the disk for it would score perfectly. This cannot be closed when the evaluator and the solver share a filesystem, so it relies on `eval/test/` being hidden from scientists and on the grader (`CONSTRAINTS.md` forbids it).
- **Supervised learning from the training labels.** `data/downloads` keeps the labelled training splits, which the visible evaluation needs. A classifier trained on them, with rotation augmentation, could label test images and build a block coupling. It is out of bounds per `DIRECTION.md` and `CONSTRAINTS.md`, and the grader must catch it. It was not run.
- **Re-downloading t10k on a login node** and shipping a lookup table inside the method. This is blatant label use; left to the grader.
- **A cue that is not a shortcut:** following the paper's preprocessing (normalise, then rotate with fill 0), rotated images have corner pixels at 0 rather than the background value. A method can use this to detect rotation. It is information present in the unlabeled data, so it was left in. Exploiting it only buys raw-OT quality on the unrotated control, not on the rotated benchmark.

## Reference score before and after

| | Rotated acc (BOT-L1, 4-dataset mean) | Unrotated acc |
| --- | ---: | ---: |
| Before: old test, half the rows of 100/class training-split instances | 0.5985 | 0.5983 |
| After: hidden test, 2 × 100/class t10k instances per experiment | 0.5915 | 0.6003 |
| After: visible train (1 × 100/class training-split instance) | 0.5870 | 0.6197 |

Per dataset (hidden test), reference vs evaluator-owned raw-pixel OT:

| Dataset | Rotated (ref / raw) | Unrotated (ref / raw) |
| --- | --- | --- |
| MNIST | 0.7215 / 0.2995 | 0.7085 / 0.8765 |
| KMNIST | 0.5095 / 0.2155 | 0.5335 / 0.8300 |
| FMNIST | 0.7070 / 0.2785 | 0.7160 / 0.7765 |
| EMNIST | 0.4279 / 0.1675 | 0.4431 / 0.6896 |

The paper's claim holds in its direction on all four datasets: BOT beats raw OT under rotation by +0.351 on average (paired SE 0.006), and trails it without rotation, as in the paper's Table 1. The paper's full-scale numbers (3,000/class) are unchanged under `_paper_results` in `scoring/reference.json`. `results` now holds the gym-scale reference.

## Noise

- The reference is deterministic: instances, rotations and the Greenkhorn solve are fixed.
- The two independent hidden replicates of each experiment give mean rotated accuracies of 0.600 and 0.583 (spread 0.016). The test averages both.
- Paired SEs of the rotated-accuracy difference are about 0.003 for small methodological changes.
- The paper's effect (0.351) is over 20× the replicate spread.
- The smallest genuine improvement found (bilinear, +0.024 rotated) is 7× its paired SE and repeats on the independent train draw.

## Headroom

- **Bounds:** the reference sits at the bound of no scored metric (`reference_at_bound = []`). Its accuracies run 0.43–0.72, mass 0.29–0.59 and margin −0.01–0.37, all against a maximum of 1.
- **Better variants:** three simple ones score above it beyond noise, and none comes near the bound (best rotated accuracy 0.615).
- **Baseline:** raw-pixel OT stays far below the reference on the rotated benchmark (0.24 vs 0.59), with plenty of room above.
- **Where the paper's own method falls short:** the unrotated control, where BOT loses 0.06–0.30 accuracy to raw pixels. It is already a weighted experiment (4 × 0.07). Now that the solver no longer knows the condition, closing that gap takes a real method, as adaptive shows. No new experiment was needed.

## Weights and coefficients

- Experiment weights are unchanged: rotated 0.18 each (the paper's headline), unrotated controls 0.07 each.
- Coefficients:
  - accuracy 1.0 (the paper's only reported quantity);
  - same-class mass 0.7 → **0.4**;
  - margin 0.4 → **0.2**.
- Why: the removed macro-accuracy (0.5) had silently doubled accuracy. Lowering the soft metrics keeps accuracy's share of the score about where it was (old effective ratio 1.5 : 1.1, now 1.0 : 0.6). The lower-regularization trial shows why that matters: sharper plans can raise mass while losing accuracy.

## Things not fixed, stated plainly

- `python validate.py` passes with one soft heuristic warning: `eval/train/evaluate.py` "regenerates the experiment" (`np.random.default_rng`). That is intended. The evaluator now builds the instances and reads the method's raw output (the couplings), which is the point of this hardening.
- The ranks above use this audit's population; the real leaderboard's per-metric spreads will differ. Signs and ratios to the paired SE are the robust part.
- The hidden test runs the solver, so a slow future method makes the test slow. No wall-clock metric or limit is imposed beyond the job's own timeout.
- The test relies on `eval/test/` (including the pool) being withheld from scientists.
- Earlier audit tooling (`redteam/rank.py`, `redteam/runner.py`, `redteam/*/run.sh`) targets the old artifact interface. It is kept only as evidence for REDTEAM.md and no longer runs against this gym.
- Full-dataset scale (about 3,000 images per class per side) is still out of reach on one CPU, so the milestone stays `near_complete`.
