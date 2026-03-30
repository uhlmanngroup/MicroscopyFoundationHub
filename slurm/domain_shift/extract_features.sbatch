#!/usr/bin/bash -l
#SBATCH --job-name=domain-shift-feats
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module --quiet load miniforge3 || module load miniforge3

PY="${PY:-$HOME/data/conda/envs/dino-peft/bin/python}"
[ -x "$PY" ] || { echo "[error] Python not found at $PY" >&2; exit 1; }

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TQDM_DISABLE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

CFG_PATH="${CFG_PATH:-configs/cluster/domain_shift/em_domain_shift_dinov2_large.yaml}"
CHECKPOINT="${CHECKPOINT:-}"

if [ ! -f "$CFG_PATH" ]; then
  echo "[error] Config file not found: $CFG_PATH" >&2
  exit 2
fi

ARGS=(--cfg "$CFG_PATH")
[ -n "$CHECKPOINT" ] && ARGS+=(--checkpoint "$CHECKPOINT")

echo "[extract-sbatch] python=$PY"
echo "[extract-sbatch] cfg=$CFG_PATH"
[ -n "$CHECKPOINT" ] && echo "[extract-sbatch] checkpoint=$CHECKPOINT"

"$PY" scripts/extract_features.py "${ARGS[@]}"
