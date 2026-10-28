#!/usr/bin/bash -l
#SBATCH --job-name=em-seg-config
#SBATCH --time=12:00:00
#SBATCH --gpus=H100:1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module --quiet load miniforge3 || module load miniforge3

PY="$HOME/data/conda/envs/dino-peft/bin/python"
[ -x "$PY" ] || { echo "Python not found at $PY" >&2; exit 1; }
: "${CFG:?Submit with --export=ALL,CFG=/absolute/path/to/runtime.yaml}"
[ -f "$CFG" ] || { echo "Config not found: $CFG" >&2; exit 1; }

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src"
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

"$PY" scripts/train_em_seg.py --cfg "$CFG"
"$PY" scripts/eval_em_seg.py --cfg "$CFG"
