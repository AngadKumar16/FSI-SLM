#!/usr/bin/env bash
# Drive the whole multi-fidelity run: high-fidelity CFD on training foils,
# rebuild, fine-tune, score, verify, regenerate figures.
#
#   ./scripts/run_multifidelity.sh                 # run every stage, resuming
#   ./scripts/run_multifidelity.sh --list          # stages and their state
#   ./scripts/run_multifidelity.sh --from cache    # start again at a stage
#   ./scripts/run_multifidelity.sh --only urans    # one stage
#   ./scripts/run_multifidelity.sh --dry-run       # print, execute nothing
#
# Each stage writes a marker in artifacts/mf_state/ and a log in artifacts/mf_logs/.
# A completed stage is skipped on the next invocation, so the script is safe to
# re-run after a crash, a reboot, or a Ctrl-C. The CFD stages are additionally
# resumable *inside* themselves: rows are flushed per case and an already-solved
# (foil, alpha) is not re-solved.
#
# The long one is `urans` -- 12 cases at hours each. Run it detached:
#   nohup ./scripts/run_multifidelity.sh > artifacts/mf_logs/driver.log 2>&1 &
#   tail -f artifacts/mf_logs/driver.log
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

PY="${PY:-python}"
MAX_WORKERS="${MAX_WORKERS:-4}"
N_FOILS="${N_FOILS:-4}"
SEED="${SEED:-0}"
CFD_DIR="${CFD_DIR:-artifacts/cfd}"
CACHE_DIR="${CACHE_DIR:-artifacts/sensor/cache}"
BASELINE="${BASELINE:-artifacts/sensor/run3/best.pt}"
RUN_DIR="${RUN_DIR:-artifacts/sensor/run_mf}"
EPOCHS="${EPOCHS:-40}"
LR="${LR:-2e-4}"
FIDELITY_WEIGHTS="${FIDELITY_WEIGHTS:-panel=1,rans=8,urans=8}"
BATCH_SIZE="${BATCH_SIZE:-16}"

STATE="artifacts/mf_state"
LOGS="artifacts/mf_logs"
mkdir -p "$STATE" "$LOGS"

STAGES=(check backup rans urans finalize verify_data cache preflight train eval verify_model figures tests)

DRY=0; ONLY=""; FROM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --list)
      for s in "${STAGES[@]}"; do
        if [ -f "$STATE/$s.done" ]; then printf '  %-12s done  (%s)\n' "$s" "$(cat "$STATE/$s.done")"
        else printf '  %-12s pending\n' "$s"; fi
      done; exit 0 ;;
    --only) ONLY="$2"; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    --redo) rm -f "$STATE/$2.done"; shift 2 ;;
    --redo-all) rm -f "$STATE"/*.done; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

_skip_until=""
[ -n "$FROM" ] && { _skip_until="$FROM"; rm -f "$STATE/$FROM.done"; }

say() { printf '\n=== %s  %s\n' "$1" "$(date '+%H:%M:%S')"; }

# Run a stage unless its marker exists. Stage output goes to its own log *and*
# to the terminal, so a detached run leaves a per-stage record.
stage() {
  local name="$1"; shift
  if [ -n "$ONLY" ] && [ "$ONLY" != "$name" ]; then return 0; fi
  if [ -n "$_skip_until" ]; then
    if [ "$_skip_until" = "$name" ]; then _skip_until=""; else return 0; fi
  fi
  if [ -z "$ONLY" ] && [ -f "$STATE/$name.done" ]; then
    echo "--- $name: already done, skipping (--redo $name to force)"; return 0
  fi
  say "$name"
  if [ "$DRY" = "1" ]; then printf '  would run: %s\n' "$*"; return 0; fi
  local t0=$SECONDS
  ( "$@" ) 2>&1 | tee "$LOGS/$name.log"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" != "0" ]; then
    echo "!!! stage '$name' failed (exit $rc). Log: $LOGS/$name.log" >&2
    exit "$rc"
  fi
  echo "$(date '+%Y-%m-%d %H:%M:%S')  ${SECONDS}s elapsed, stage took $((SECONDS - t0))s" \
    > "$STATE/$name.done"
}

# --------------------------------------------------------------------------
# check: fail now, not four hours in
# --------------------------------------------------------------------------
do_check() {
  local bad=0
  for cmd in gmsh; do
    command -v "$cmd" >/dev/null || { echo "MISSING binary: $cmd"; bad=1; }
  done
  if command -v openfoam2606 >/dev/null; then
    openfoam2606 -c 'which pimpleFoam simpleFoam' || { echo "MISSING solver in openfoam2606"; bad=1; }
  else
    echo "MISSING binary: openfoam2606 (brew install --cask openfoam via the gerlero tap)"; bad=1
  fi
  "$PY" - <<'PYEOF' || bad=1
import importlib, sys
missing = [m for m in ("numpy", "torch", "scipy", "matplotlib", "gmsh")
           if importlib.util.find_spec(m) is None]
print("python deps missing:", missing or "none")
sys.exit(1 if missing else 0)
PYEOF
  [ -f "$BASELINE" ] || { echo "MISSING baseline checkpoint: $BASELINE"; bad=1; }
  echo "--- disk free"; df -h "$ROOT" | tail -1
  echo "--- the plan (nothing is spent yet)"
  "$PY" cli.py sensor-multifidelity --n-foils "$N_FOILS" --seed "$SEED" --out-dir "$CFD_DIR"
  [ "$bad" = "0" ] || { echo "preflight failed; fix the above before spending CFD hours"; return 1; }
  echo "OK"
}

# --------------------------------------------------------------------------
# backup: finalize rewrites manifest.jsonl and splits.json in place
# --------------------------------------------------------------------------
do_backup() {
  local dst="$CFD_DIR/backup_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$dst"
  for f in manifest.jsonl splits.json summary.json; do
    [ -f "$CFD_DIR/$f" ] && cp -p "$CFD_DIR/$f" "$dst/"
  done
  cp -p artifacts/sensor/eval_test_panel.json "$dst/" 2>/dev/null || true
  echo "backed up to $dst"
  ls -la "$dst"
}

# --------------------------------------------------------------------------
# CFD. Steady below stall, unsteady above it; both resume.
# --------------------------------------------------------------------------
do_rans()  { "$PY" cli.py sensor-multifidelity --run --steady-only \
               --n-foils "$N_FOILS" --seed "$SEED" \
               --out-dir "$CFD_DIR" --max-workers "$MAX_WORKERS"; }

do_urans() { "$PY" cli.py sensor-multifidelity --run --unsteady-only \
               --n-foils "$N_FOILS" --seed "$SEED" \
               --out-dir "$CFD_DIR" --max-workers "$MAX_WORKERS"; }

do_finalize() { "$PY" cli.py sensor-finalize --out-dir "$CFD_DIR"; }

# --------------------------------------------------------------------------
# Gates. verify_data is the one that decides whether training is legitimate.
# --------------------------------------------------------------------------
do_verify_data() { "$PY" scripts/verify_multifidelity.py data \
                    --cfd-dir "$CFD_DIR" --n-foils "$N_FOILS" --seed "$SEED"; }

do_cache() { "$PY" cli.py sensor-cache --manifest "$CFD_DIR/manifest.jsonl" \
               --splits "$CFD_DIR/splits.json" --out-dir "$CACHE_DIR"; }

do_preflight() { "$PY" cli.py sensor-preflight --cache-dir "$CACHE_DIR" \
                   --epochs "$EPOCHS" --batch-size "$BATCH_SIZE"; }

do_train() {
  "$PY" cli.py sensor-train \
    --cache-dir "$CACHE_DIR" --run-dir "$RUN_DIR" \
    --init-from "$BASELINE" --lr "$LR" --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" --fidelity-weights "$FIDELITY_WEIGHTS"
}

do_eval() {
  local ck="$RUN_DIR/best.pt"
  for f in panel rans urans; do
    echo "--- test / $f"
    "$PY" cli.py sensor-eval --checkpoint "$ck" --cache-dir "$CACHE_DIR" \
      --split test --fidelity "$f" --converged-only \
      --out-path "artifacts/sensor/eval_test_${f}_mf.json" || \
      echo "(no $f rows in the test split -- expected for urans until the sweep covers a test foil)"
  done
  echo "--- val / all fidelities"
  "$PY" cli.py sensor-eval --checkpoint "$ck" --cache-dir "$CACHE_DIR" \
    --split val --out-path artifacts/sensor/eval_val_mf.json
  echo "--- baseline on the same rows, for the comparison"
  "$PY" cli.py sensor-eval --checkpoint "$BASELINE" --cache-dir "$CACHE_DIR" \
    --split test --fidelity rans --converged-only \
    --out-path artifacts/sensor/eval_test_rans_baseline.json
}

do_verify_model() { "$PY" scripts/verify_multifidelity.py model \
                      --cfd-dir "$CFD_DIR" --run-dir "$RUN_DIR" --baseline "$BASELINE"; }

do_figures() {
  "$PY" -m sensor.figures --out poster/generated
  # The slide comparing fidelities has to plot the fine-tuned run, not the
  # baseline it was measured against.
  if [ -f artifacts/sensor/eval_test_rans_mf.json ]; then
    ( cd poster/figure_scripts && \
      FSI_EVAL_PANEL=artifacts/sensor/eval_test_panel_mf.json \
      FSI_EVAL_RANS=artifacts/sensor/eval_test_rans_mf.json \
      "$PY" fig_slides_rans.py )
  else
    echo "no fine-tuned RANS eval; leaving the fidelity slide on baseline numbers"
  fi
}

do_tests() { "$PY" -m pytest tests/ -q; }

stage check        do_check
stage backup       do_backup
stage rans         do_rans
stage urans        do_urans
stage finalize     do_finalize
stage verify_data  do_verify_data
stage cache        do_cache
stage preflight    do_preflight
stage train        do_train
stage eval         do_eval
stage verify_model do_verify_model
stage figures      do_figures
stage tests        do_tests

say "all stages complete"
echo "Numbers to copy into docs/results.md and the poster:"
echo "  artifacts/sensor/eval_test_{panel,rans,urans}_mf.json"
echo "  $LOGS/verify_model.log"
