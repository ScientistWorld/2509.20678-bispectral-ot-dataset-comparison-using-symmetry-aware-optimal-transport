# Scoring audit

Run `bash redteam/run_all.sh` inside the gym container as a CPU job. Each named
folder has a `run.sh` that produces its variant or shortcut, calls the visible
train evaluator with that manifest, and invokes the original hidden test with
**no arguments** in a separate Python process. It saves the resulting scores in
that folder. No evaluator, Python environment, or evaluator process is modified.

The audit orchestrator may reuse the reference coupling artifacts, as permitted
by the audit request. The shortcut builders that synthesize artifacts use only
visible data, preprocessing, and the manifest interface. They never import the
test evaluator or read test answers. The reference snapshot is only for auditing
and ranking, not an input to those builders.

`rank.py` reports the requested weighted sum of standardized improvements with
absolute coefficient magnitudes and `higher_is_better` directions. The exact
future attempt population is unknowable; it records the finite audit population
used for normalization, also reports unstandardized differences, and distinguishes
componentwise dominance from population-dependent improvements.
