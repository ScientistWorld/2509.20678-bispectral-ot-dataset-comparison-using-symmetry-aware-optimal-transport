# Train/Test Evaluation Design Notes

## Checks Performed

- Read `scripts/evaluate_train.sh`, `scripts/evaluate_test.sh`, `eval/train/evaluate.py`, and `eval/test/evaluate.py` side by side.
- Confirmed the split evaluators read saved method artifacts (`.npz` couplings) and do not import from `method/`.
- Checked for train/test cross-imports and copy-paste path errors with `rg`.
- Checked that `scoring/scores_train.json` and `scoring/scores_test.json` are absent.
- Ran `bash -n` on both split scripts.
- Checked `scoring/reference.json` metric entries for required `higher_is_better`, `coefficient`, and `description` fields.

## Changes Made

- Strengthened metric triangulation in both split evaluators:
  - kept class-preservation accuracy and macro class preservation;
  - added `same_class_mass_mean` to reward total transported mass landing on same-label targets;
  - added `class_mass_margin_mean` to distinguish barely-correct argmax couplings from robust same-class transport;
  - added `target_effective_support_ratio` to penalize collapsed couplings that concentrate mass on too few target samples;
  - added `nonzero_source_row_fraction` to catch artifacts that leave evaluated source rows without outgoing mass.
- Updated `scoring/reference.json` metric definitions for all existing experiments without changing paper result values or experiment weights.
- Removed the hardcoded zero-argument train default of `outputs/bot/manifest.json`.
- Added zero-argument manifest discovery under `outputs/*/manifest.json` for the train split.
- Added deterministic duplicate handling for discovery mode while keeping explicit manifest arguments strict.
- Updated the test split fallback so it discovers manifests if the train artifact index is missing, instead of falling back to a single hardcoded manifest.

## Anti-Shortcut Rationale

The first-pass split already used a useful stratified source-row half split, so a method tuned to visible source rows can regress on held-out rows. The added metrics make the visible target harder to game: a method now has to preserve same-class mass continuously, maintain margin against wrong classes, cover targets broadly, and keep source transport valid. These are all computed from the same saved artifact interface, so future methods remain free to use different internals as long as they emit compatible coupling artifacts.

## Runtime Concerns For Next Iteration

- I did not run the evaluators or inspect generated numbers in this design-only pass.
- The split evaluators now emit metrics that the top-level `eval/evaluate_artifacts.py` does not emit. This pass was scoped to `eval/train/` and `eval/test/`; the next runner should ensure the train/test scoring path, not the legacy single-score path, is the one used for split validation.
- Discovery mode keeps the last duplicate experiment/method entry in sorted manifest order. This avoids old output folders breaking zero-argument evaluation, but explicit manifest lists still error on duplicates. The next runner should verify that the discovered artifacts are the intended final-scale artifacts.
- `target_effective_support_ratio` is a collapse check, not a paper headline metric. Watch whether it saturates near 1.0 for all valid OT artifacts; if it does, keep it as a validity guard with low weight or replace it with a sharper coupling-quality constraint.
