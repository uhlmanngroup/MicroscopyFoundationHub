#!/usr/bin/bash -l
#SBATCH --job-name=dino-single-paired
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:H100:1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module --quiet load gpu   || module load gpu
module --quiet load mamba || module load mamba

PY="$HOME/data/conda/envs/dino-peft/bin/python"
[ -x "$PY" ] || { echo "Python not found at $PY"; exit 1; }

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs runs

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src"
export TQDM_DISABLE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MLFLOW_TRACKING_URI="file:/scratch/$USER/mlruns_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"

BASE_CFG="config/dinopoc_lora_cluster.yaml"
DINO_SIZE=${DINO_SIZE:-base}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-2}
NUM_WORKERS=${NUM_WORKERS:-4}
AMP=${AMP:-false}
RUN_TAG=${RUN_TAG:-paired${SLURM_JOB_ID:-manual}}

prepare_cfg() {
  local base_cfg="$1" size="$2" epochs="$3" batch="$4" workers="$5" amp="$6" tag="$7"
  "$PY" - "$base_cfg" "$size" "$epochs" "$batch" "$workers" "$amp" "$tag" <<'PY'
import sys, yaml, tempfile, pathlib
base, size, epochs, batch, workers, amp, tag = sys.argv[1:]
cfg = yaml.safe_load(open(base))
cfg["dino_size"] = size
cfg["epochs"] = max(1, int(epochs))
cfg["batch_size"] = max(1, int(batch))
cfg["num_workers"] = max(0, int(workers))
cfg["amp"] = amp.lower() not in ("0","false","no","off")
tag = tag or "paired"
out_dir = pathlib.Path(cfg["out_dir"]).expanduser() / tag
cfg["out_dir"] = str(out_dir)
ml_name = cfg.get("mlflow_experiment_name", "paired-run")
cfg["mlflow_experiment_name"] = f"{ml_name}_{tag}"
tmp = tempfile.NamedTemporaryFile(prefix=f"{tag}_", suffix=".yaml", delete=False)
with open(tmp.name, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(tmp.name)
print(cfg["out_dir"])
PY
}

mapfile -t prep < <(prepare_cfg "$BASE_CFG" "$DINO_SIZE" "$EPOCHS" "$BATCH_SIZE" "$NUM_WORKERS" "$AMP" "$RUN_TAG")
RUNTIME_CFG="${prep[0]}"
RUN_DIR="${prep[1]}"

echo "[single] cfg=$RUNTIME_CFG  run_dir=$RUN_DIR"
trap 'rm -f "$RUNTIME_CFG"' EXIT

"$PY" scripts/train_em_seg.py --cfg "$RUNTIME_CFG"
"$PY" scripts/eval_em_seg.py --cfg "$RUNTIME_CFG" --out_csv "$RUN_DIR/${RUN_TAG}_metrics.csv"
