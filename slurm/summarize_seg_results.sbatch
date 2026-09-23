#!/usr/bin/bash -l
#SBATCH --job-name=seg-summary
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module --quiet load miniforge3 || module load miniforge3

PY="$HOME/data/conda/envs/dino-peft/bin/python"
[ -x "$PY" ] || { echo "[error] Python not found at $PY" >&2; exit 1; }

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# dino_peft_path(): machine paths straight from configs/paths.yaml
source slurm/lib/common.sh
mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src"
export TQDM_DISABLE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

ROOT=${ROOT:-"$(dino_peft_path scratch_root)/dinov3"}
# The script takes a folder NAME, created under ROOT — not a path. Passing
# --summary-dir here used to make argparse reject the whole call.
SUMMARY_DIR_NAME=${SUMMARY_DIR_NAME:-summary}
EXTRA_ARGS=(--summary-dir-name "$SUMMARY_DIR_NAME")

echo "[summary] root=$ROOT"
echo "[summary] reports -> $ROOT/$SUMMARY_DIR_NAME"

"$PY" scripts/analysis/summarize_seg_results.py --root "$ROOT" "${EXTRA_ARGS[@]}"
