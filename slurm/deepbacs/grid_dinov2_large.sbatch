#!/usr/bin/bash -l
#SBATCH --job-name=deepbacs-aureus-dinov2l-frozen
#SBATCH --time=12:00:00
#SBATCH --gpus=H100:1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
module --quiet load miniforge3 || module load miniforge3

PY="$HOME/data/conda/envs/dino-peft/bin/python"
[ -x "$PY" ] || { echo "Python not found at $PY"; exit 1; }

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs runs

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src"
export TQDM_DISABLE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

BASE_CFG=${BASE_CFG:-configs/cluster/deepbacs/deepbacs_single_dinov2_cluster.yaml}
DINO_SIZE=${DINO_SIZE:-large}
BACKBONE_NAME=${BACKBONE_NAME:-dinov2}
EPOCHS=${EPOCHS:-}
BATCH_SIZE=${BATCH_SIZE:-}
NUM_WORKERS=${NUM_WORKERS:-}
AMP=${AMP:-}
REPEATS=${REPEATS:-5}
BASE_SEED=${BASE_SEED:-1}
BASE_SPLIT_SEED=${BASE_SPLIT_SEED:-101}
RUN_PREFIX=${RUN_PREFIX:-deepbacs-aureus-dinov2large-frozen}

archive_slurm_log() {
  local dest_dir="$1" tag="$2"
  local src="${SLURM_SUBMIT_DIR:-$PWD}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
  if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    local array_src="${SLURM_SUBMIT_DIR:-$PWD}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
    if [ -f "$array_src" ]; then
      src="$array_src"
    fi
  fi
  if [ -f "$src" ]; then
    local dest="${dest_dir}/logs/${tag}_slurm.out"
    mkdir -p "$(dirname "$dest")"
    mv "$src" "$dest"
    echo "[log] moved slurm log -> $dest"
  else
    echo "[log] slurm output not found at $src" >&2
  fi
}

prepare_cfg() {
  local base_cfg="$1" size="$2" epochs="$3" batch="$4" workers="$5" amp="$6" seed="$7" split_seed="$8" tag="$9" backbone_name="${10}"
  "$PY" - "$base_cfg" "$size" "$epochs" "$batch" "$workers" "$amp" "$seed" "$split_seed" "$tag" "$backbone_name" <<'PY'
import pathlib
import re
import sys
import tempfile
import yaml

base, size, epochs, batch, workers, amp, seed, split_seed, tag, backbone_name = sys.argv[1:]
cfg = yaml.safe_load(open(base))
cfg["dino_size"] = size
backbone = cfg.get("backbone") or {}
backbone["name"] = backbone_name or backbone.get("name", "dinov2")
backbone["variant"] = size
cfg["backbone"] = backbone

if seed:
    cfg["seed"] = int(seed)
if split_seed:
    cfg["split_seed"] = int(split_seed)
if epochs:
    cfg["epochs"] = max(1, int(epochs))
if batch:
    cfg["batch_size"] = max(1, int(batch))
if workers:
    cfg["num_workers"] = max(0, int(workers))
if amp:
    cfg["amp"] = amp.lower() not in ("0", "false", "no", "off")

tag = tag or "manual"
task_type = cfg.get("task_type", "seg")
default_out = pathlib.Path(cfg.get("out_dir", "runs/exp")).expanduser()
exp_base = cfg.get("experiment_id") or default_out.name
cfg["experiment_id"] = f"{exp_base}_{tag}"
root = pathlib.Path(cfg.get("results_root", default_out.parent)).expanduser()
cfg["results_root"] = str(root)
modality = str(cfg.get("modality", "em")).strip().lower() or "em"
out_dir = root / modality / task_type / cfg["experiment_id"]
cfg["out_dir"] = str(out_dir)

tmp = tempfile.NamedTemporaryFile(prefix=f"{tag}_", suffix=".yaml", delete=False)
with open(tmp.name, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(tmp.name)
print(cfg["out_dir"])
PY
}

run_rep() {
  local rep_idx="$1"
  local seed=$((BASE_SEED + rep_idx))
  local split_seed=$((BASE_SPLIT_SEED + rep_idx))
  local rep_tag="${RUN_PREFIX}_rep$((rep_idx + 1))"
  mapfile -t prep < <(prepare_cfg "$BASE_CFG" "$DINO_SIZE" "$EPOCHS" "$BATCH_SIZE" "$NUM_WORKERS" "$AMP" "$seed" "$split_seed" "$rep_tag" "$BACKBONE_NAME")
  local runtime_cfg="${prep[0]}"
  local run_dir="${prep[1]}"
  echo "[rep=$((rep_idx + 1))] seed=$seed split_seed=$split_seed cfg=$runtime_cfg"
  "$PY" scripts/train_em_seg.py --cfg "$runtime_cfg"
  "$PY" scripts/eval_em_seg.py --cfg "$runtime_cfg" --out_csv "$run_dir/${rep_tag}_metrics.csv"
  archive_slurm_log "$run_dir" "$rep_tag"
  rm -f "$runtime_cfg"
}

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  task_id=${SLURM_ARRAY_TASK_ID}
  if (( task_id < 0 || task_id >= REPEATS )); then
    echo "[error] SLURM_ARRAY_TASK_ID=$task_id outside 0..$((REPEATS - 1))" >&2
    exit 1
  fi
  run_rep "$task_id"
else
  echo "[info] SLURM_ARRAY_TASK_ID not set; running $REPEATS reps sequentially"
  for ((rep=0; rep<REPEATS; rep++)); do
    run_rep "$rep"
  done
fi
