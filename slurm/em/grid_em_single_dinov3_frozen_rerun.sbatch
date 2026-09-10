#!/usr/bin/bash -l
#SBATCH --job-name=em-single-dinov3l-frozen
#SBATCH --time=12:00:00
#SBATCH --gpus=H100:1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%A_%a.out

# -----------------------------------------------------------------------
# EM single-dataset frozen rerun — DINOv3-L/16, head only, 5 varied seeds
# Fixes the seed=0 bug in the original single-dataset frozen runs (std=0).
#
# Required env var:
#   DATASET   droso | kasthuri | lucchi
#
# Usage:
#   bash slurm/em/submit_em_single_frozen_rerun.sh
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

DATASET=${DATASET:-droso}
REPEATS=${REPEATS:-5}
BASE_SEED=${BASE_SEED:-1}
BASE_SPLIT_SEED=${BASE_SPLIT_SEED:-101}
WEIGHTS="/home/cfuste/scratch/data/models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
RESULTS_ROOT="/home/cfuste/scratch/DINO-LoRA/dinov3/seg"

case "$DATASET" in
  droso)
    BASE_CFG="configs/cluster/EM/droso_dinov3_cluster.yaml"
    TRAIN_IMG="/home/cfuste/data/datasets/Droso/Train_In"
    TEST_IMG="/home/cfuste/data/datasets/Droso/Test_In"
    ;;
  kasthuri)
    BASE_CFG="configs/cluster/EM/kasthuri_dinov3_head_cluster.yaml"
    TRAIN_IMG="/home/cfuste/data/datasets/Kasthuri++/Train_In"
    TEST_IMG="/home/cfuste/data/datasets/Kasthuri++/Test_In"
    ;;
  lucchi)
    BASE_CFG="configs/cluster/EM/lucchi_dinov3_head_cluster.yaml"
    TRAIN_IMG="/home/cfuste/data/datasets/Lucchi++/Train_In"
    TEST_IMG="/home/cfuste/data/datasets/Lucchi++/Test_In"
    ;;
  *)
    echo "[error] Unknown DATASET='$DATASET'. Use: droso | kasthuri | lucchi" >&2
    exit 1
    ;;
esac

archive_slurm_log() {
  local dest_dir="$1" tag="$2"
  local src="${SLURM_SUBMIT_DIR:-$PWD}/logs/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
  [ -f "$src" ] || src="${SLURM_SUBMIT_DIR:-$PWD}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
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
  local base_cfg="$1" seed="$2" split_seed="$3" tag="$4" dataset="$5"
  local train_img="$6" test_img="$7" weights="$8" results_root="$9"
  "$PY" - "$base_cfg" "$seed" "$split_seed" "$tag" "$dataset" \
          "$train_img" "$test_img" "$weights" "$results_root" <<'PY'
import pathlib, sys, tempfile, yaml

(base_cfg, seed, split_seed, tag, dataset,
 train_img, test_img, weights, results_root) = sys.argv[1:]

cfg = yaml.safe_load(open(base_cfg))

# frozen mode
cfg["full_finetune"] = False
cfg["use_lora"]      = False
if "lora" in cfg:
    cfg["lora"]["enabled"] = False
cfg["batch_size"] = 2

# backbone — fix to vitl16 + scratch weights
backbone = cfg.get("backbone", {})
backbone["variant"] = "vitl16"
backbone["weights"] = weights
cfg["backbone"] = backbone
cfg["dino_size"] = "vitl16"

# data paths
cfg["train_img_dir"] = train_img
cfg["test_img_dir"]  = test_img
# preserve mask dirs if same convention (Train_In → Train_Out)
cfg["train_mask_dir"] = train_img.replace("Train_In", "Train_Out")
cfg["test_mask_dir"]  = test_img.replace("Test_In", "Test_Out")

# results to scratch
cfg["results_root"] = results_root

# seeds
if seed:       cfg["seed"]       = int(seed)
if split_seed: cfg["split_seed"] = int(split_seed)

# identifiers
tag = tag or "manual"
cfg["task_type"]     = f"{dataset}-nolora"
cfg["experiment_id"] = f"2026-09-10_em_{dataset}_dinov3_frozen_seg_{tag}"
root = pathlib.Path(results_root)
out_dir = root / "em" / cfg["task_type"] / cfg["experiment_id"]
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
  local rep_tag="rep$((rep_idx + 1))"

  mapfile -t prep < <(prepare_cfg \
    "$BASE_CFG" "$seed" "$split_seed" "$rep_tag" "$DATASET" \
    "$TRAIN_IMG" "$TEST_IMG" "$WEIGHTS" "$RESULTS_ROOT")

  local runtime_cfg="${prep[0]}"
  local run_dir="${prep[1]}"
  echo "[em-single dataset=$DATASET rep=$((rep_idx + 1))] seed=$seed split_seed=$split_seed"
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
  echo "[info] running $REPEATS reps sequentially (dataset=$DATASET)"
  for ((rep=0; rep<REPEATS; rep++)); do
    run_rep "$rep"
  done
fi
