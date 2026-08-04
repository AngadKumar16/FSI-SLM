#!/usr/bin/env bash
# Double-click this in Finder. macOS opens Terminal and runs it -- no typing.
#
# It runs the check stage first and stops if the environment is not ready, so a
# double-click can never start a day of CFD on a machine that is missing gmsh.
cd "$(dirname "$0")/.."

echo "FSI-SLM multi-fidelity run"
echo "working in: $PWD"
echo

./scripts/run_multifidelity.sh --only check || {
  echo
  echo "Environment check failed -- nothing was run. See artifacts/mf_logs/check.log"
  echo "Press return to close."
  read -r _
  exit 1
}

echo
echo "Check passed. The full run is 20 steady + 12 unsteady CFD cases,"
echo "then a fine-tune and the evals. Expect a day or more."
printf 'Start it? [y/N] '
read -r reply
case "$reply" in
  y|Y|yes|YES) ;;
  *) echo "Nothing started."; echo "Press return to close."; read -r _; exit 0 ;;
esac

./scripts/run_multifidelity.sh
echo
echo "Done. Logs in artifacts/mf_logs/. Press return to close."
read -r _
