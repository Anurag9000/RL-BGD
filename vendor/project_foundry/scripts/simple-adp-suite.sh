#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPOSITORY_ROOT"

if [[ -x "$REPOSITORY_ROOT/.venv/bin/python" ]]; then
  PYTHON="$REPOSITORY_ROOT/.venv/bin/python"
elif [[ -x "$REPOSITORY_ROOT/.venv/Scripts/python.exe" ]]; then
  PYTHON="$REPOSITORY_ROOT/.venv/Scripts/python.exe"
else
  PYTHON="$(command -v python3 || command -v python || printf 'python')"
fi

export PYTHONPATH="$REPOSITORY_ROOT/src:$REPOSITORY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
CPU_CORES="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')"
for variable in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS GOTO_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS TORCH_NUM_THREADS; do
  if [[ -z "${!variable:-}" ]]; then
    printf -v "$variable" '%s' "$CPU_CORES"
    export "$variable"
  fi
done
export TORCH_INTEROP_THREADS="${TORCH_INTEROP_THREADS:-1}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-ACTIVE}"
export TABULAR_CHILD_SHARED_CPU="${TABULAR_CHILD_SHARED_CPU:-0}"
export TABULAR_CPU_WORKERS="${TABULAR_CPU_WORKERS:-0}"
renice -20 -p "$$" >/dev/null 2>&1 || true
ionice -c2 -n0 -p "$$" >/dev/null 2>&1 || true
chrt -b -p 0 "$$" >/dev/null 2>&1 || true
exec "$PYTHON" -u -m simple_adp.runner.launcher "$@"
