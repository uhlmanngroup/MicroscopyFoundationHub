#!/usr/bin/bash -l
#SBATCH --job-name=missing-summary
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%x-%j.out

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/lib/common.sh"
dino_peft_init
: "${RUN_TAG:?RUN_TAG is required}"
"$PY" scripts/analysis/summarize_missing_20261020.py --run-tag "$RUN_TAG"
