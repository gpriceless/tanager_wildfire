#!/usr/bin/env bash
# Execute the five deliverable notebooks sequentially under a hard memory bound.
#
# Why this wrapper exists
# -----------------------
# The notebooks used to be executed by calling `jupyter nbconvert --execute`
# directly, several at once. Nothing bounded that: each notebook holds multiple
# full 426-band cubes, so concurrent runs exhausted RAM and the OOM killer took
# down unrelated long-running services alongside the run.
#
# Two guards prevent a repeat:
#
#   1. Sequential execution. tests/test_notebooks.py parametrizes the notebooks,
#      so pytest runs them one at a time and peak memory is one notebook rather
#      than five. It also pins TANAGER_MAX_JOBS=2 and renders to a temp dir, so
#      the repo copies are left untouched.
#
#   2. A transient systemd scope with its own MemoryMax. A notebook that blows
#      the budget is killed inside its own cgroup and nothing else on the machine
#      is affected — verified: the scope has been OOM-killed repeatedly while
#      unrelated services stayed up.
#
# Sizing the bound
# ----------------
# Measured peaks, sequential, TANAGER_MAX_JOBS=2:
#
#   01-data-discovery     passes
#   02-burn-severity      6.2 GB   (was >14 GB before the raw cubes were released)
#   03-fuel-moisture      exceeds 15 GB — compute_lfmc_indices upcasts the whole
#                         cube to float64; currently fails
#   04-temporal-recovery  exceeds 11 GB — holds four masked scenes at once before
#                         reprojecting; currently fails
#   05-sensor-comparison  passes
#
# So the default 12G is deliberately *below* what 03 and 04 currently need: those
# two are expected to fail until their memory use is fixed, and a bound that hides
# that would defeat the point. Note this means the suite does not pass end to end
# today. Raise it for a one-off via NOTEBOOK_SUITE_MEM_MAX, but leave headroom —
# the box has 31 GB total and other services need to keep running.
#
# Usage:  scripts/run_notebook_suite.sh [extra pytest args...]
#   e.g.  scripts/run_notebook_suite.sh -k burn-severity
#         NOTEBOOK_SUITE_MEM_MAX=16G scripts/run_notebook_suite.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MEM_MAX="${NOTEBOOK_SUITE_MEM_MAX:-12G}"
LOG_DIR="${REPO_ROOT}/outputs/notebook-suite"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/run-$(date +%Y%m%d-%H%M%S).log"

if [[ ! -x .venv/bin/pytest ]]; then
  echo "error: .venv/bin/pytest not found — create the venv first" >&2
  exit 1
fi

if ! command -v systemd-run >/dev/null 2>&1; then
  echo "error: systemd-run not found; refusing to run unbounded" >&2
  exit 1
fi

echo "notebook suite: sequential execution under MemoryMax=${MEM_MAX}"
echo "log: ${LOG_FILE}"
echo

# MemorySwapMax=0 makes the bound bite immediately rather than degrading into
# swap thrash. pipefail propagates pytest's exit status through tee.
systemd-run --user --scope --quiet \
  --unit="notebook-suite-$$" \
  -p MemoryMax="${MEM_MAX}" \
  -p MemorySwapMax=0 \
  -- .venv/bin/pytest tests/test_notebooks.py -m slow -v --no-header "$@" \
  2>&1 | tee "$LOG_FILE"
