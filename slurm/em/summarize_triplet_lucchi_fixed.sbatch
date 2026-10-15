#!/usr/bin/bash -l
#SBATCH --job-name=em-triplet-fixed-summary
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/lib/common.sh"
dino_peft_init
: "${RUN_TAG:?RUN_TAG must identify the corrected rerun}"
"$PY" scripts/analysis/summarize_triplet_lucchi_fixed.py --run-tag "$RUN_TAG"
