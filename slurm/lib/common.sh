# shellcheck shell=bash
#
# Shared setup for every sweep script. Source it right after the #SBATCH block:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
#     dino_peft_init
#
# Then describe one repetition in a `run_rep` function and hand it to
# `dino_peft_dispatch`, which runs a single array task or, outside an array,
# every repetition in sequence.

set -euo pipefail

# Interpreter used for every python call. Override by exporting PY.
: "${PY:=$HOME/data/conda/envs/dino-peft/bin/python}"

dino_peft_init() {
  module --quiet load miniforge3 2>/dev/null || module load miniforge3 2>/dev/null || true

  if [ ! -x "$PY" ]; then
    echo "[error] python not found at '$PY'. Export PY=/path/to/python." >&2
    exit 1
  fi

  cd "${SLURM_SUBMIT_DIR:-$PWD}"
  mkdir -p logs

  export PYTHONUNBUFFERED=1
  export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
  export TQDM_DISABLE=1
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
}

# dino_peft_path <key>
# Echo one entry from configs/paths.yaml (honouring DINO_PEFT_* overrides), so
# sweep scripts read machine paths from the same place the configs do.
dino_peft_path() {
  "$PY" -c "
import sys
sys.path.insert(0, 'src')
from dino_peft.config import load_paths
print(load_paths()['$1'])
"
}

# archive_slurm_log <run_dir> <tag>
# Move this task's SLURM output next to the run it produced, so a result
# directory carries its own log.
archive_slurm_log() {
  local dest_dir="$1" tag="$2"
  local log_dir="${SLURM_SUBMIT_DIR:-$PWD}/logs"
  local src="${log_dir}/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-0}.out"

  if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    for candidate in \
      "${log_dir}/${SLURM_JOB_NAME:-job}-${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID}.out" \
      "${log_dir}/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-}_${SLURM_ARRAY_TASK_ID}.out"; do
      [ -f "$candidate" ] && { src="$candidate"; break; }
    done
  fi

  if [ -f "$src" ]; then
    mkdir -p "${dest_dir}/logs"
    mv "$src" "${dest_dir}/logs/${tag}_slurm.out"
    echo "[log] moved slurm log -> ${dest_dir}/logs/${tag}_slurm.out"
  else
    echo "[log] slurm output not found at $src" >&2
  fi
}

# make_runtime_cfg <args...>
# Thin wrapper over scripts/utils/make_runtime_cfg.py. Sets RUNTIME_CFG and
# RUN_DIR in the caller's scope.
make_runtime_cfg() {
  local output
  output="$("$PY" scripts/utils/make_runtime_cfg.py "$@")"
  RUNTIME_CFG="$(printf '%s\n' "$output" | sed -n 1p)"
  RUN_DIR="$(printf '%s\n' "$output" | sed -n 2p)"
  if [ -z "$RUNTIME_CFG" ] || [ -z "$RUN_DIR" ]; then
    echo "[error] make_runtime_cfg produced no config path" >&2
    return 1
  fi
}

# train_and_eval <runtime_cfg> <run_dir> <tag>
# Train, evaluate, archive the log, then drop the throwaway config.
train_and_eval() {
  local runtime_cfg="$1" run_dir="$2" tag="$3"
  "$PY" scripts/train_em_seg.py --cfg "$runtime_cfg"
  "$PY" scripts/eval_em_seg.py  --cfg "$runtime_cfg" --out_csv "${run_dir}/${tag}_metrics.csv"
  archive_slurm_log "$run_dir" "$tag"
  rm -f "$runtime_cfg"
}

# dino_peft_dispatch <repeats> <fn>
# Inside a job array, run only this task's repetition; otherwise run them all.
# <fn> is called with the zero-based repetition index.
dino_peft_dispatch() {
  local repeats="$1" fn="$2"
  if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    local task_id=${SLURM_ARRAY_TASK_ID}
    if (( task_id < 0 || task_id >= repeats )); then
      echo "[error] SLURM_ARRAY_TASK_ID=$task_id outside 0..$((repeats - 1))" >&2
      exit 1
    fi
    "$fn" "$task_id"
  else
    echo "[info] no job array; running $repeats repetitions sequentially"
    for ((rep = 0; rep < repeats; rep++)); do
      "$fn" "$rep"
    done
  fi
}
