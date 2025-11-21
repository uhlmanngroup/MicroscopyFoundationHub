#!/usr/bin/bash -l
#SBATCH --job-name=dino-grid-lucchi
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:H100:1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MLFLOW_TRACKING_URI="file:/scratch/$USER/mlruns_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"

BASE_CFG="config/lucchi_lora_cluster.yaml"
DINO_SIZES=${DINO_SIZES:-"small base large giant"}
EPOCHS=${EPOCHS:-5}
BATCH_DEFAULT=${BATCH_DEFAULT:-2}
NUM_WORKERS=${NUM_WORKERS:-4}
AMP=${AMP:-false}
RUN_TAG=${RUN_TAG:-lucchigrid${SLURM_JOB_ID:-manual}}

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
tag = tag or "grid"
out_dir = pathlib.Path(cfg["out_dir"]).expanduser() / tag
cfg["out_dir"] = str(out_dir)
ml_name = cfg.get("mlflow_experiment_name", "lucchi-grid")
cfg["mlflow_experiment_name"] = f"{ml_name}_{tag}"
tmp = tempfile.NamedTemporaryFile(prefix=f"{tag}_", suffix=".yaml", delete=False)
with open(tmp.name, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(tmp.name)
print(cfg["out_dir"])
PY
}

get_batch_for_size() {
  local size_key="${1^^}"
  local var="BATCH_${size_key}"
  local val="${!var:-}"
  if [ -n "$val" ]; then
    echo "$val"
  else
    echo "$BATCH_DEFAULT"
  fi
}

idx=0
for size in $DINO_SIZES; do
  idx=$((idx + 1))
  size_clean="${size// /}"
  size_tag="${RUN_TAG}_s${idx}_${size_clean}"
  batch_val="$(get_batch_for_size "$size_clean")"
  mapfile -t prep < <(prepare_cfg "$BASE_CFG" "$size_clean" "$EPOCHS" "$batch_val" "$NUM_WORKERS" "$AMP" "$size_tag")
  runtime_cfg="${prep[0]}"
  run_dir="${prep[1]}"
  echo "[grid/$idx] size=$size_clean batch=$batch_val cfg=$runtime_cfg"
  "$PY" scripts/train_em_seg.py --cfg "$runtime_cfg"
  "$PY" scripts/eval_em_seg.py --cfg "$runtime_cfg" --out_csv "$run_dir/${size_tag}_metrics.csv"
  rm -f "$runtime_cfg"
done
