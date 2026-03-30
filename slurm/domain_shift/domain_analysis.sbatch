#!/usr/bin/bash -l
#SBATCH --job-name=domain-shift-analysis
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

CFG_PATH="${CFG_PATH:-configs/cluster/domain_shift/domain_shift_3way_dinov2_large.yaml}"

if [ ! -f "$CFG_PATH" ]; then
  echo "[error] Config file not found: $CFG_PATH" >&2
  exit 2
fi

echo "[domain-analysis-sbatch] python=$PY"
echo "[domain-analysis-sbatch] cfg=$CFG_PATH"

"$PY" scripts/run_domain_analysis.py --cfg "$CFG_PATH"
