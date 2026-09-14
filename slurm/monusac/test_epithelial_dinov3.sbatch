#!/usr/bin/bash -l
#SBATCH --job-name=monusac-smoketest
#SBATCH --time=00:30:00
#SBATCH --gpus=H100:1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out

# Smoke test: MoNuSAC Epithelial, DINOv3-L/16, head-only, 5 epochs.
# Run with: sbatch slurm/monusac/test_epithelial_dinov3.sbatch

set -euo pipefail
module --quiet load miniforge3 || module load miniforge3

PY="$HOME/data/conda/envs/dino-peft/bin/python"
[ -x "$PY" ] || { echo "Python not found at $PY"; exit 1; }

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src"
export TQDM_DISABLE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

CFG="configs/cluster/monusac/monusac_epithelial_dinov3_test.yaml"

echo "[smoketest] train..."
"$PY" scripts/train_em_seg.py --cfg "$CFG"

echo "[smoketest] eval..."
RUN_DIR=$("$PY" - "$CFG" <<'PY'
import yaml, pathlib, sys
cfg = yaml.safe_load(open(sys.argv[1]))
root = pathlib.Path(cfg["results_root"])
out = root / cfg.get("modality","em") / cfg["task_type"] / cfg["experiment_id"]
print(out)
PY
)
"$PY" scripts/eval_em_seg.py --cfg "$CFG" --out_csv "${RUN_DIR}/smoketest_metrics.csv"

echo "[smoketest] done — results at: $RUN_DIR"

# Archive log
LOG_SRC="${SLURM_SUBMIT_DIR:-$PWD}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
[ -f "$LOG_SRC" ] && mv "$LOG_SRC" "${RUN_DIR}/logs/smoketest_slurm.out" || true
