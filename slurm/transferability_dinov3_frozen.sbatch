#!/usr/bin/bash -l
#SBATCH --job-name=dinov3l-transfer
#SBATCH --time=12:00:00
#SBATCH --gpus=H100:1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/%x-%A_%a.out

# Array 0-29: 2 modalities x 3 source datasets x 5 trained heads.
# Each task performs inference on the other two target test datasets.

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/lib/common.sh"
dino_peft_init

ARGS=(--modality both)
[ -n "${TRANSFER_RUNS_ROOT:-}" ] && ARGS+=(--runs-root "$TRANSFER_RUNS_ROOT")
[ -n "${TRANSFER_DATA_ROOT:-}" ] && ARGS+=(--data-root "$TRANSFER_DATA_ROOT")
[ -n "${TRANSFER_OUT_DIR:-}" ] && ARGS+=(--out-dir "$TRANSFER_OUT_DIR")
[ "${TRANSFER_INCLUDE_DIAGONAL:-0}" = 1 ] && ARGS+=(--include-diagonal)
"$PY" scripts/eval_transferability.py \
  --job-index "${SLURM_ARRAY_TASK_ID:?submit with --array=0-29}" \
  "${ARGS[@]}"
