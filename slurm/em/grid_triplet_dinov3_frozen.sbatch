#!/usr/bin/bash -l
#SBATCH --job-name=em-triplet-dinov3l-frozen
#SBATCH --time=12:00:00
#SBATCH --gpus=H100:1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%A_%a.out

# -----------------------------------------------------------------------
# EM triplet (VNC + Lucchi++ + Kasthuri++) — DINOv3-L/16 frozen (head only)
# Reruns all 5 reps with proper varied seeds (original runs all used seed=0).
#
# Usage:
#   bash slurm/em/submit_em_triplet_frozen.sh
# -----------------------------------------------------------------------

set -euo pipefail
module --quiet load miniforge3 || module load miniforge3

PY="$HOME/data/conda/envs/dino-peft/bin/python"
[ -x "$PY" ] || { echo "Python not found at $PY"; exit 1; }

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src"
export TQDM_DISABLE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

BASE_CFG="configs/cluster/EM/paired_dinov3_cluster.yaml"
REPEATS=${REPEATS:-5}
BASE_SEED=${BASE_SEED:-1}
BASE_SPLIT_SEED=${BASE_SPLIT_SEED:-101}

archive_slurm_log() {
  local dest_dir="$1" tag="$2"
  local src="${SLURM_SUBMIT_DIR:-$PWD}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
  if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    local array_src="${SLURM_SUBMIT_DIR:-$PWD}/logs/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
    [ -f "$array_src" ] && src="$array_src"
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
  local base_cfg="$1" seed="$2" split_seed="$3" tag="$4"
  "$PY" - "$base_cfg" "$seed" "$split_seed" "$tag" <<'PY'
import pathlib, sys, tempfile, yaml

base_cfg, seed, split_seed, tag = sys.argv[1:]
cfg = yaml.safe_load(open(base_cfg))

cfg["full_finetune"] = False
cfg["use_lora"]      = False
if "lora" in cfg:
    cfg["lora"]["enabled"] = False
cfg["batch_size"]  = 2

if seed:       cfg["seed"]       = int(seed)
if split_seed: cfg["split_seed"] = int(split_seed)

tag = tag or "manual"
cfg["task_type"]     = "triplet-nolora"
cfg["experiment_id"] = f"2026-09-09_em_triplet_dinov3_frozen_seg_{tag}"
root = pathlib.Path(cfg.get("results_root", "runs")).expanduser()
cfg["results_root"]  = str(root)
out_dir = root / "em" / cfg["task_type"] / cfg["experiment_id"]
cfg["out_dir"]       = str(out_dir)

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
  local rep_tag="rep$((rep_idx + 1))"

  mapfile -t prep < <(prepare_cfg "$BASE_CFG" "$seed" "$split_seed" "$rep_tag")

  local runtime_cfg="${prep[0]}"
  local run_dir="${prep[1]}"
  echo "[em-triplet frozen rep=$((rep_idx + 1))] seed=$seed split_seed=$split_seed"
  "$PY" scripts/train_em_seg.py --cfg "$runtime_cfg"
  "$PY" scripts/eval_em_seg.py  --cfg "$runtime_cfg" --out_csv "$run_dir/${rep_tag}_metrics.csv"
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
  echo "[info] running $REPEATS reps sequentially (frozen)"
  for ((rep=0; rep<REPEATS; rep++)); do
    run_rep "$rep"
  done
fi
