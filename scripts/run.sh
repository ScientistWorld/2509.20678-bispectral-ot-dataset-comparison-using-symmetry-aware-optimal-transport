#!/bin/bash
# Hardening verification job (iteration 11). One CPU job, resumable: every
# finished step leaves redteam/hardening/results/<name>/<side>.done and is
# skipped on rerun. Cost matrices are memoised in outputs/cost_cache (identical
# numbers with or without it) so variants share the expensive distances.

set -uo pipefail
cd /home/user
export BOT_COST_CACHE=/home/user/outputs/cost_cache
export PYTHONUNBUFFERED=1
H=redteam/hardening
R=$H/results
mkdir -p "$R" "$BOT_COST_CACHE"
echo "[run] start $(date)"

run_side() {  # side name solver [extra evaluator args...]
  local side=$1 name=$2 solver=$3; shift 3
  local d=$R/$name
  if [ -f "$d/$side.done" ]; then echo "[skip] $name/$side"; return 0; fi
  mkdir -p "$d"
  local t0; t0=$(date +%s)
  local extra=()
  if [ "$side" = train ]; then extra=(--confusion-out "$d/confusion_matrices_train.json"); fi
  HARDENING_LOG=$d/solver_log_$side.json python -m eval.$side.evaluate --solver "$solver" \
    --out "$d/scores_$side.json" --status-out "$d/status_$side.json" \
    --details-out "$d/details_$side.json" "${extra[@]}" "$@" > "$d/eval_$side.txt" 2>&1
  local rc=$?
  echo "{\"returncode\": $rc, \"seconds\": $(( $(date +%s) - t0 ))}" > "$d/run_$side.json"
  touch "$d/$side.done"
  echo "[done] $name/$side rc=$rc $(( $(date +%s) - t0 ))s $(date +%T)"
}

# 0. Gate: the evaluator must accept a valid-but-uninformative plan and reject
#    an all-zero one before hours are spent on the expensive solvers.
run_side test uniform "$H/solvers/uniform.sh"
run_side test zero "$H/solvers/zero.sh"
if ! grep -q '"status": "ok"' "$R/uniform/status_test.json" || ! grep -q '"status": "failed"' "$R/zero/status_test.json"; then
  echo "[gate] evaluator sanity check FAILED"; tail -40 "$R/uniform/eval_test.txt" "$R/zero/eval_test.txt"; exit 2
fi
echo "[gate] evaluator sanity check passed"

# 1. Reference = the paper's BOT through the real entry points (scripts/solve.sh).
if [ ! -f "$R/reference/test.done" ]; then
  mkdir -p "$R/reference"; t0=$(date +%s)
  bash scripts/evaluate_test.sh --details-out "$R/reference/details_test.json" \
    --baselines-out "$R/reference/baselines_test.json" > "$R/reference/eval_test.txt" 2>&1
  rc=$?
  cp scoring/scores_test.json "$R/reference/scores_test.json"; cp scoring/test_status.json "$R/reference/status_test.json"
  echo "{\"returncode\": $rc, \"seconds\": $(( $(date +%s) - t0 ))}" > "$R/reference/run_test.json"
  touch "$R/reference/test.done"; echo "[done] reference/test rc=$rc $(( $(date +%s) - t0 ))s"
fi
if ! grep -q '"status": "ok"' "$R/reference/status_test.json"; then
  echo "[gate] reference failed on the hidden test"; tail -60 "$R/reference/eval_test.txt"; rm -f "$R/reference/test.done"; exit 3
fi
if [ ! -f "$R/reference/train.done" ]; then
  t0=$(date +%s)
  bash scripts/evaluate_train.sh --details-out "$R/reference/details_train.json" \
    --baselines-out "$R/reference/baselines_train.json" > "$R/reference/eval_train.txt" 2>&1
  rc=$?
  cp scoring/scores_train.json "$R/reference/scores_train.json"; cp scoring/train_status.json "$R/reference/status_train.json"
  cp scoring/confusion_matrices_train.json "$R/reference/confusion_matrices_train.json"
  echo "{\"returncode\": $rc, \"seconds\": $(( $(date +%s) - t0 ))}" > "$R/reference/run_train.json"
  touch "$R/reference/train.done"; echo "[done] reference/train rc=$rc $(( $(date +%s) - t0 ))s"
fi

# 2. Shortcuts replayed through the hidden test (cheap).
for s in nan signed tiny empty order_blocks nn_assignment train_lookup; do
  run_side test "$s" "$H/solvers/$s.sh"
done
run_side train train_lookup "$H/solvers/train_lookup.sh"
# Forged scores: the solver overwrites scoring/*.json itself; the evaluator must overwrite them back.
if [ ! -f "$R/forged_extras/test.done" ]; then
  mkdir -p "$R/forged_extras"
  bash scripts/evaluate_test.sh --solver "$H/solvers/forged_extras.sh" --details-out "$R/forged_extras/details_test.json" \
    > "$R/forged_extras/eval_test.txt" 2>&1
  echo "{\"returncode\": $?}" > "$R/forged_extras/run_test.json"
  cp scoring/scores_test.json "$R/forged_extras/scores_test.json"; cp scoring/test_status.json "$R/forged_extras/status_test.json"
  touch "$R/forged_extras/test.done"; echo "[done] forged_extras/test"
fi

# 3. Honest variants (better / weaker) on the hidden test, cheapest first.
run_side test adaptive "$H/solvers/adaptive.sh"
run_side test raw_geometry "$H/solvers/raw_geometry.sh"
run_side test bilinear "$H/solvers/bilinear.sh"
run_side test combined "$H/solvers/combined.sh"
run_side test lower_reg "$H/solvers/lower_reg.sh"
run_side test bot_l22 "$H/solvers/bot_l22.sh"

# 4. Same variants on the visible train instances (train/test consistency).
run_side train adaptive "$H/solvers/adaptive.sh"
run_side train raw_geometry "$H/solvers/raw_geometry.sh"
run_side train bilinear "$H/solvers/bilinear.sh"
run_side train combined "$H/solvers/combined.sh"

# 5. Final: the hidden test exactly as the system runs it (no arguments), then
#    restore the reference score files and validate.
bash scripts/evaluate_test.sh > "$R/final_noarg_test.txt" 2>&1
echo "{\"returncode\": $?}" > "$R/final_noarg_test.json"
python - <<'PY'
import json
a = json.load(open("scoring/scores_test.json")); b = json.load(open("redteam/hardening/results/reference/scores_test.json"))
json.dump({"identical_to_reference_run": a == b}, open("redteam/hardening/results/final_noarg_check.json", "w"))
print("final no-arg test identical to reference run:", a == b)
PY
cp "$R/reference/scores_train.json" scoring/scores_train.json
cp "$R/reference/scores_train.json" scoring/scores.json
cp "$R/reference/status_train.json" scoring/train_status.json
cp "$R/reference/confusion_matrices_train.json" scoring/confusion_matrices_train.json
python redteam/hardening/build_reference.py --scores scoring/scores_test.json --baselines "$R/reference/baselines_test.json"
python -m redteam.hardening.analyze > "$H/analysis.txt" 2>&1; cat "$H/analysis.txt"
python validate.py > "$H/validate.txt" 2>&1; tail -25 "$H/validate.txt"
du -sh outputs/cost_cache
echo "[run] end $(date)"
