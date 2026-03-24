#!/usr/bin/bash -l
#SBATCH --job-name=deepbacs-paired-balance
#SBATCH --time=24:00:00
#SBATCH --gpus=H100:1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=0-4

set -euo pipefail
module --quiet load miniforge3 || module load miniforge3

PY="$HOME/data/conda/envs/dino-peft/bin/python"
[ -x "$PY" ] || { echo "Python not found at $PY"; exit 1; }

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src"
export TQDM_DISABLE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

SEED="${SLURM_ARRAY_TASK_ID:-0}"
CFG="${CFG:-configs/cluster/deepbacs/deepbacs_paired_balanced_dinov2_cluster.yaml}"
VARIANT="${VARIANT:-both}"
TUNING_MODE="${TUNING_MODE:-cfg}"
EPOCHS="${EPOCHS:-}"

echo "[deepbacs-balanced] seed=${SEED} cfg=${CFG} variant=${VARIANT} tuning_mode=${TUNING_MODE}"

CMD=(
  "$PY" scripts/ablation_deepbacs_paired_balance.py
  --cfg "$CFG"
  --seed "$SEED"
  --variant "$VARIANT"
  --tuning-mode "$TUNING_MODE"
)

if [ -n "$EPOCHS" ]; then
  CMD+=(--epochs "$EPOCHS")
fi

"${CMD[@]}"
