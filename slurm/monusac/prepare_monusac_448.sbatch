#!/usr/bin/bash -l
#SBATCH --job-name=monusac-prep448
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out

# Preprocess MoNuSAC patches to uniform 448x448.
# Run with: sbatch slurm/monusac/prepare_monusac_448.sbatch

set -euo pipefail
module --quiet load miniforge3 || module load miniforge3

PY="$HOME/data/conda/envs/dino-peft/bin/python"
[ -x "$PY" ] || { echo "Python not found at $PY"; exit 1; }

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export PYTHONPATH="$PWD/src"

IN_ROOT="${IN_ROOT:-/home/cfuste/data/datasets/monusac}"
OUT_ROOT="${OUT_ROOT:-/home/cfuste/data/datasets/monusac_448}"

echo "[prep] Input:  $IN_ROOT"
echo "[prep] Output: $OUT_ROOT"

"$PY" scripts/data/prepare_monusac_448.py \
    --in-root "$IN_ROOT" \
    --out-root "$OUT_ROOT"
