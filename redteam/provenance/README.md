# Compute job evidence

- `93a3b3c0-770/`: original batched audit, including genuine feature and OT
  recomputation, reference seeds 0/1/2, and shortcut cases. Job exit code 0.
  Scheduler preemptions resumed saved progress; some Greenkhorn runs reached
  the iteration limit. Intermediate rankings and the initial stale empty-case
  fallback in this log are superseded by the follow-up and final summary.
- `2cfcdbd5-aa5/`: follow-up job, exit code 0. Selected the actual empty
  artifacts (both evaluators reject them), ran the reconstructed-test shortcut
  (train accuracy 0, test 1), and scored the genuine combined improvement.

The original scheduler logs and metadata are copied here so the audit can
restore its newly created top-level `jobs/` entries to the initial git state.
Log extensions are `.txt` so these specific small evidence files are tracked;
ordinary per-attempt evaluator log spam remains ignored.

The final rank population and results are in `../summary.json`, with divisors
in `../rank_coordinates.json`. The root `REDTEAM.md` explains corrections and
limits. No progress-log rank should replace the final rank.
