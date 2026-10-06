#!/usr/bin/bash -l
#SBATCH --job-name=joint-metric
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Submit through `python scripts/joint_metric.py submit`; the CLI selects GPU
# resources only for extraction and places all logs in persistent result storage.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/lib/common.sh
dino_peft_init
MODELS="$(dino_peft_path models_root)"
SCRATCH="$(dino_peft_path scratch_root)"
export HF_HOME="${HF_HOME:-$MODELS/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$MODELS/torch}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRATCH/joint_metric/runtime/matplotlib-${SLURM_JOB_ID:-local}}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export MPLBACKEND=Agg
mkdir -p "$HF_HOME" "$TORCH_HOME" "$MPLCONFIGDIR"
exec "$PY" scripts/joint_metric.py "$@"
