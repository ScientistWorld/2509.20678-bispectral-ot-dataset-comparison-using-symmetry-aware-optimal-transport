# Does this gym reward better research?

**Partly: genuine quality changes move the score correctly, but major shortcuts beat the reference.** A legitimate condition-aware geometry improvement scores **+1.649596** relative to the reference, and removing rotation-invariant geometry scores **−3.206436**. Both exceed the three-seed score spread of **0.603372**. The bilinear representation fix alone gains only **+0.390606**, within that spread; its rotated improvement is inconclusive. Public-answer shortcuts obtain perfect metrics, and an invalid coupling with 10% matching accuracy ranks above every genuine method in the audit population.

## Genuine variants and quality sensitivity

All variants retain the four datasets, 100 samples per class in each half, both rotation conditions, all four ground costs, and all 64 result entries. Accuracies below average the 16 proposed-method entries in each condition; the rank includes **every** scored raw and proposed entry. The unrotated condition is explicitly known in the protocol, so choosing its geometry does not use labels or change the input condition.

| Variant and executable evidence | Actual change | Test rank Δ vs reference | Rotated accuracy | Unrotated accuracy |
| --- | --- | ---: | ---: | ---: |
| [Reference](redteam/reference/run.sh) | Original saved couplings, rescored | 0.000000 | 0.547808 | 0.549317 |
| [Better combined](redteam/better_combined/run.sh) | Centered bilinear polar sampling for rotated BOT; raw geometry for the unrotated control | **+1.649596** | 0.568692 | **0.803250** |
| [Better control geometry](redteam/better_control_geometry/run.sh) | Keep rotated BOT; use raw geometry only for the ordinary unrotated condition | **+1.378137** | 0.547808 | **0.803250** |
| [Bilinear fix](redteam/better_bilinear/run.sh) | Match the image rotation center and interpolate polar coordinates; keep the bispectrum and OT solver | +0.390606 | 0.568692 | 0.570202 |
| [Lower regularization trial](redteam/better_lower_reg/run.sh) | Actual BOT solve with regularization 0.003 instead of 0.01 | −1.209787 | 0.515981 | 0.517808 |
| [Weaker raw geometry](redteam/weaker_raw_geometry/run.sh) | Remove the bispectral representation; use pixel costs in the proposed-method slots | **−3.206436** | **0.222231** | 0.803250 |

The combined variant's robust gain comes from preserving ordinary matching: unrotated accuracy improves by 0.253933, compared with seed spread 0.011481. Its rotated gain of 0.020885 is smaller than rotated seed spread 0.028000. Thus this audit demonstrates sensitivity to a real improvement in the scored control, and to removing the central idea; it does **not** establish a rotated-only advance beyond noise. Reducing regularization was an honest unsuccessful trial and correctly scores lower. Some original and variant Greenkhorn solves hit the iteration limit; all use the reference solver budget, and the numerical warnings are retained in the job evidence.

### How rank was computed

For each experiment, result key, and metric, the reported Δ is the sum of

`experiment_weight × abs(coefficient) × better_direction × (attempt − reference) / population_std`.

Directions come from `higher_is_better`; the negative coefficient on `label_cost_used` contributes its magnitude, with lower values better. Rotated experiments have weight 0.18 each, and unrotated controls 0.07 each. Coefficients are accuracy 1, macro accuracy 0.5, same-class mass 0.7, class-mass margin 0.4, target support 0.2, and label-cost flag magnitude 1. Counts have coefficient zero. Constant coordinates contribute zero. There is no averaging away the raw comparison entries.

Normalization uses the **20 finite completed audit attempts including the reference**, listed in [summary.json](redteam/summary.json). The exact future population is unavailable, so these are reproducible audit-population ranks, not a prediction of an unseen leaderboard. [rank.py](redteam/rank.py) implements the calculation, and [rank_coordinates.json](redteam/rank_coordinates.json) records every divisor. NaN and failed attempts are excluded. Perfect-metric shortcuts improve 320 scored coordinates and tie the remaining 64; their dominance over the reference does not depend on this population. Mixed tradeoffs, including the signed shortcut, can depend on it. Raw unstandardized totals are retained for inspection but are **not** used as ranks.

## Seed spread, reproduction, and headroom

The reference was recomputed with source/target split seeds 0, 1, and 2, paired with rotation seeds 1, 2, and 3. Each recomputation runs the reference bispectral features, raw costs, all four distances, and Greenkhorn solves. This measures sampling and rotation variability; the algorithm is deterministic given those seeds.

| Recomputed reference | Test rank Δ | Rotated accuracy | Unrotated accuracy |
| --- | ---: | ---: | ---: |
| [Seed 0](redteam/reference_seed_0/run.sh) | 0.000000 | 0.547808 | 0.549317 |
| [Seed 1](redteam/reference_seed_1/run.sh) | +0.098407 | 0.540385 | 0.556808 |
| [Seed 2](redteam/reference_seed_2/run.sh) | +0.603372 | 0.568385 | 0.560798 |
| **Maximum minus minimum** | **0.603372** | **0.028000** | **0.011481** |

Seed 0 reproduces all original test metrics within **7.8 × 10⁻¹⁶**. Gains below the observed score range are treated as noise. [Per-coordinate seed spreads](redteam/reference/seed_spread_by_metric.json) preserve variation that the means hide.

**The central reported direction reproduces:** BOT beats raw OT in all 16 dataset/distance rotated comparisons. L1 examples from the actual hidden test:

| Dataset | Raw rotated | BOT rotated | Raw unrotated | BOT unrotated |
| --- | ---: | ---: | ---: | ---: |
| MNIST | 0.298000 | 0.710000 | 0.880000 | 0.702000 |
| KMNIST | 0.194000 | 0.558000 | 0.852000 | 0.566000 |
| Fashion-MNIST | 0.238000 | 0.686000 | 0.796000 | 0.692000 |
| EMNIST letters | 0.173077 | 0.440000 | 0.700769 | 0.433077 |

These scaled test accuracies do not reproduce the full-data paper magnitudes recorded in the gym's reference table (BOT L1: 0.8405, 0.7815, 0.7617, 0.5983). Ordinary unrotated matching is materially worse than raw OT on every dataset; the control exposes that sacrifice. The paper itself was not used by the audit or the shortcut builders.

**The reference has headroom.** Its only scored coordinates already at the ideal bound are `label_cost_used = 0` in **all 64 result entries**: the Cartesian product of four datasets `{mnist, kmnist, fmnist, emnist}`, two experiments `{rotated_matching, unrotated_baseline}`, and eight keys `{raw_ot_l1, raw_ot_l2, raw_ot_l22, raw_ot_cos, bispectral_ot_l1, bispectral_ot_l2, bispectral_ot_l22, bispectral_ot_cos}`. [bounds.json](redteam/reference/bounds.json) enumerates every coordinate. No accuracy, macro accuracy, same-class mass, margin, or target-support coordinate is at its valid optimum. The flag is hard-coded to zero by evaluation, so its apparent saturation provides no constraint enforcement. The simple raw baseline is far below BOT under rotations (mean 0.222231 vs 0.547808), with substantial room above the reference. Uniform and zero couplings also score clearly below it.

## Shortcuts tested

Each row links to a real standalone `run.sh`; its folder contains the manifest, evaluator return codes, and produced train/test scores. Reference rank is **0**. Every shortcut was passed to the original hidden test with **no arguments**. No hidden evaluator code, process, or Python environment was modified. All successful cases supply all 64 original result keys.

| Shortcut | What it does | Test rank Δ vs reference | Result |
| --- | --- | ---: | --- |
| [Shortcut: visible labels](redteam/shortcut_visible_labels/run.sh) | Build a block coupling directly from public source/target labels. Even declares `label_cost_used=1` inside the artifact. | **+20.532995** | **Works.** All five benefit metrics become 1; evaluator emits label-cost flag 0. |
| [Shortcut: order answers](redteam/shortcut_order_answers/run.sh) | Match source and target indices by `index // 100`, exploiting the generator's contiguous class blocks. No image features enter the coupling. | **+20.532995** | **Works.** Perfect test metrics with the original labels and full sizes. |
| [Shortcut: rebuild test answers](redteam/shortcut_rebuild_test_answers/run.sh) | Complement the public train splitter; use the order shortcut only on those rows and deliberately assign every train row to the wrong class. | **+20.532995** | **Works.** Train accuracy 0, hidden test accuracy 1 in every result. |
| [Shortcut: forged truth](redteam/shortcut_forged_truth/run.sh) | Keep full array sizes, replace all truth labels by an invented alternating binary partition, and return its content-independent coupling. | **+20.532995** | **Works.** Evaluator accepts the replacement truth and reports perfect metrics. |
| [Shortcut: tiny test](redteam/shortcut_tiny_test/run.sh) | Replace each side by two samples per class and supply their label-block coupling. | **+20.532995** | **Works.** Test sources shrink from 500/1300 to 10/26; counts have zero weight and no size check. |
| [Shortcut: remove rotation](redteam/shortcut_remove_rotation/run.sh) | Submit saved unrotated raw-pixel couplings under both condition names and every result key. | **+11.399857** | **Works.** Reported rotated accuracy becomes 0.793413 without solving the rotated task. |
| [Shortcut: replace baseline artifact](redteam/shortcut_replace_baseline/run.sh) | Replace only the supposedly fixed raw comparator artifacts with the reference's BOT artifacts; keep every proposed-method artifact unchanged. | **+2.446508** | **Works.** Rank improves with no improvement to the proposed method. |
| [Shortcut: signed failed matching](redteam/shortcut_signed_failed_matching/run.sh) | Correct just 10% of rows, using positive mass 1,000,000 and negative mass −999,999 across classes on those rows. | **+26.921036** | **Works in the audit population.** Accuracy is 0.1 everywhere, but impossible mass/margin dominate the aggregate. |
| [Shortcut: score fields / reference replay](redteam/shortcut_score_fields/run.sh) | Reuse unchanged reference artifacts; overwrite score files with perfect scores, negative label flags and timings, and success fields. | **0.000000** | **Tie shortcut pays by replay.** Direct score forgery fails: hidden evaluation recomputes every metric and ignores the extra fields. |
| [Shortcut: uniform output](redteam/shortcut_uniform/run.sh) | Constant uniform coupling with real labels and full sizes. | −14.008755 | **Fails to pay.** Accuracy averages 0.084615 despite optimal target support. |
| [Shortcut: zero output](redteam/shortcut_zero/run.sh) | All-zero coupling with real labels and full sizes. | −21.495768 | **Fails to pay.** Accepted as an artifact but scores below reference. |
| [Shortcut: empty output](redteam/shortcut_empty/run.sh) | Zero source rows; explicitly select that artifact for the no-argument hidden test after train rejection. | No score | **Fails.** Both evaluators reject it with `No source rows selected`. |
| [Shortcut: NaN output](redteam/shortcut_nan/run.sh) | Full-size NaN coupling. | Undefined | **No verified paying rank.** Both evaluators exit 0 and serialize NaN metrics; this is a validation gap, not a demonstrated leaderboard gain. |

### Why a failing task can outrank the reference

The signed shortcut has accuracy and macro accuracy 0.1 in every experiment. Yet the evaluator accepts negative coupling entries, row-normalizes them, and reports same-class mass approximately **100,000** and margin **99,999.1**, quantities that should be bounded by 1 for valid probability mass. Its balanced construction also preserves nearly uniform target totals, earning support approximately 1. The **standardized** rank contributions are accuracy −9.706050, macro accuracy −4.853025, mass +25.694598, margin +14.682648, and support +1.102865: total **+26.921036**. This directly fails the requested task-failure check even after per-metric standardization; the claim is limited to the recorded population.

The perfect-metric shortcuts are stronger population-independent evidence: every scored benefit improves and the label-use penalty never changes. The evaluator derives ground truth, class vocabulary, sample counts, and the held-out row split from agent-authored NPZ arrays. It does not bind them to actual images or verify either condition. Public labels and class ordering already disclose the matching answers. Public split code with seed 1729 also exposes which half will be tested. Replacing comparator artifacts is rewarded because baseline result keys enter the same positive improvement sum as the proposed method. No additional fixed pretrained model is present; the comparator coupling artifacts are the fixed-side asset tested here.

## Execution and retained evidence

Heavy feature computation, transport solves, artifact generation, and all evaluator runs executed in CPU jobs **93a3b3c0-770** and **2cfcdbd5-aa5**, both exit code 0. The second job confirmed the corrected empty-artifact rejection, reconstructed-test shortcut, and combined genuine variant using saved outputs. [Job evidence](redteam/provenance/README.md) retains logs and metadata. Bundling used saved files and separate evaluator subprocesses; no shortcut relies on a process surviving into the test. A fresh-container test can reproduce each shortcut from its saved manifest/artifacts or rebuild them with its `run.sh`.

The initial empty-output run accidentally evaluated the stale reference artifact index after train failure. Its apparent tie was discarded, preserved only in explicitly named `stale_*` evidence, and retested with the empty artifact selected. Intermediate job-log ranks also used unfinished populations; the final [summary](redteam/summary.json) supersedes them.

Shortcut construction uses public data, preprocessing, visible train split code, and the public artifact interface. Existing reference couplings were reused where the audit request permits it; the runner regenerates those outputs from the supplied method if absent. Synthetic shortcut builders never read hidden test scores, test code, or paper contents. The reference-score snapshot and rank specification are used only for audit comparisons. The test always runs as `bash scripts/evaluate_test.sh` with no arguments, after installing each attempt's artifacts through visible evaluation. Directly written score fields are tested only in the explicitly named score-field shortcut.

Every variant's large artifacts and intermediate costs are reproducible and gitignored under `redteam/`; scores and executable source are tracked. The temporary `scripts/run.sh` and scoring outputs are restored to the initial revision with git. The reproduction milestone remains `near_complete`; this iteration completes the scoring audit rather than changing the reproduced method.

For the next gym iteration, evaluation should own image identities, truth, sample counts, transformations, and a truly withheld input split; validate finite nonnegative couplings and prescribed marginals; and recompute or lock comparator results outside the mutable artifact interface. A constant self-reported label flag cannot enforce unsupervised matching. Validity constraints should be checked before the weighted scientific score is calculated.
