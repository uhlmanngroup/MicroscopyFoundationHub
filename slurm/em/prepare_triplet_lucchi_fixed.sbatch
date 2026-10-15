#!/usr/bin/bash -l
#SBATCH --job-name=em-triplet-fix-data
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/lib/common.sh"
dino_peft_init

EM_ROOT="$(dino_peft_path data_root)/EM"
SOURCE="$EM_ROOT/TRIPLET"
OUTPUT="$EM_ROOT/TRIPLET_LUCCHI_FIXED"
LUCCHI="$EM_ROOT/Lucchi++"

if [ ! -e "$OUTPUT" ]; then
  "$PY" scripts/data/rebuild_triplet_lucchi.py \
    --triplet-root "$SOURCE" --lucchi-root "$LUCCHI" --output-root "$OUTPUT"
fi

"$PY" scripts/data/validate_triplet_lucchi_fixed.py \
  --dataset-root "$OUTPUT" --lucchi-root "$LUCCHI" --write-mapping
