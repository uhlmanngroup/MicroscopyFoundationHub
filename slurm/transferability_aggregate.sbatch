#!/usr/bin/bash -l
#SBATCH --job-name=dinov3l-transfer-summary
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%x-%j.out

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/lib/common.sh"
dino_peft_init

ARGS=(--modality both)
[ -n "${TRANSFER_RUNS_ROOT:-}" ] && ARGS+=(--runs-root "$TRANSFER_RUNS_ROOT")
[ -n "${TRANSFER_OUT_DIR:-}" ] && ARGS+=(--out-dir "$TRANSFER_OUT_DIR")
[ "${TRANSFER_INCLUDE_DIAGONAL:-0}" = 1 ] && ARGS+=(--include-diagonal)
"$PY" scripts/eval_transferability.py --aggregate-only \
  "${ARGS[@]}"
