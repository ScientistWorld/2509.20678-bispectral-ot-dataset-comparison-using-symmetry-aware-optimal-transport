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
Snapshotting private score/rank metadata is optional audit provenance; absence
of those files in an agent-visible workspace does not block shortcut builders.

`rank.py` reports the requested weighted sum of standardized improvements with
absolute coefficient magnitudes and `higher_is_better` directions. The exact
future attempt population is unknowable; it records the finite audit population
used for normalization, also reports unstandardized differences, and distinguishes
componentwise dominance from population-dependent improvements.

`REDTEAM.md` at the workspace root is the final report. `summary.json` records
the final 20-member finite population; `rank_coordinates.json` records every
normalization divisor. The first job's intermediate ranks used an incomplete
population and are superseded by these final files.

`shortcut_empty` fails both evaluators. Its explicitly named `stale_*` files
preserve a discarded initial harness result that accidentally selected the
old reference artifact index after training failed. The follow-up selected the
empty artifacts and confirmed rejection; no stale result is counted as a tie.

Each standalone `run.sh` rebuilds its shortcut or resumes the expensive genuine
variant's saved couplings, then reruns both evaluators. `run_all.sh` additionally
sets `AUDIT_RESUME=1` to skip completed cases after job preemption. Large NPZ
artifacts and cost caches are reproducible and ignored by git; scores, manifests,
run scripts, outcomes, and final ranking evidence are tracked.
