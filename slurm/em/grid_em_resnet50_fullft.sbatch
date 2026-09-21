#!/usr/bin/bash -l
#SBATCH --job-name=em-resnet50-unet-fullft
#SBATCH --time=24:00:00
#SBATCH --gpus=H100:1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%A_%a.out

# -----------------------------------------------------------------------
# CNN baseline — UNet + ImageNet-supervised ResNet50, end-to-end fine-tune.
# Null-hypothesis control for the saliency claim (ISBI 2026): does joint EM
# training collapse Lucchi++ for a CNN, as it does for DINOv3-L/16?
#
# Seeds are deliberately IDENTICAL to the DINOv3 EM runs (seed 1-5,
# split_seed 101-105) so the comparison is paired: same train/val partition
# per rep, which removes one variance source from the DINOv3-minus-CNN delta.
#
# Usage:
#   bash slurm/em/submit_em_resnet50_fullft.sh          # all 4 datasets
#   sbatch --array=0-4 --export=ALL,DATASET=triplet \
#          slurm/em/grid_em_resnet50_fullft.sbatch      # one cell
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

DATASET=${DATASET:-lucchi}
REPEATS=${REPEATS:-5}
BASE_SEED=${BASE_SEED:-1}         # matches the DINOv3 EM runs: seeds 1-5
BASE_SPLIT_SEED=${BASE_SPLIT_SEED:-101}   # ... and split_seeds 101-105
BATCH_SIZE=${BATCH_SIZE:-}        # empty -> use the config's own value
EPOCHS=${EPOCHS:-}                # empty -> use the config's own value (smoke tests: EPOCHS=2)

case "$DATASET" in
  lucchi)   BASE_CFG="configs/cluster/EM/lucchi_resnet50_fullft_cluster.yaml" ;;
  kasthuri) BASE_CFG="configs/cluster/EM/kasthuri_resnet50_fullft_cluster.yaml" ;;
  droso)    BASE_CFG="configs/cluster/EM/droso_resnet50_fullft_cluster.yaml" ;;
  triplet)  BASE_CFG="configs/cluster/EM/triplet_resnet50_fullft_cluster.yaml"
            # DINOv3 triplet fullft ran at batch_size 1; match it.
            BATCH_SIZE=${BATCH_SIZE:-1} ;;
  *) echo "[error] unknown DATASET='$DATASET' (lucchi|kasthuri|droso|triplet)" >&2; exit 1 ;;
esac
[ -f "$BASE_CFG" ] || { echo "[error] config not found: $BASE_CFG" >&2; exit 1; }

archive_slurm_log() {
  local dest_dir="$1" tag="$2"
  local src="${SLURM_SUBMIT_DIR:-$PWD}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
  if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    local array_src="${SLURM_SUBMIT_DIR:-$PWD}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
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
  local base_cfg="$1" seed="$2" split_seed="$3" tag="$4" dataset="$5" batch="$6" epochs="$7"
  "$PY" - "$base_cfg" "$seed" "$split_seed" "$tag" "$dataset" "$batch" "$epochs" <<'PY'
import pathlib, sys, tempfile, yaml

sys.path.insert(0, "src")
from dino_peft.config import expand_placeholders, load_paths

base_cfg, seed, split_seed, tag, dataset, batch, epochs = sys.argv[1:]
cfg = yaml.safe_load(open(base_cfg))

cfg["full_finetune"] = True
cfg["use_lora"]      = False
if batch:
    cfg["batch_size"] = int(batch)
if epochs:
    cfg["epochs"] = int(epochs)

if seed:       cfg["seed"]       = int(seed)
if split_seed: cfg["split_seed"] = int(split_seed)

tag = tag or "manual"
base_exp = cfg.get("experiment_id") or f"{dataset}_resnet50_fullft"
cfg["experiment_id"] = f"{base_exp}_{tag}"

# The config spells its roots as ${results_root}/${scratch_root}; resolve them the
# way load_config() will, because run_dir is echoed back to bash, which would not
# expand a ${...} of its own and would make a directory by that literal name.
paths = load_paths()
root = pathlib.Path(expand_placeholders(cfg.get("results_root", "runs"), paths)).expanduser()
cfg["results_root"] = str(root)
modality = str(cfg.get("modality", "em")).strip().lower() or "em"
out_dir = root / modality / cfg["task_type"] / cfg["experiment_id"]
cfg["out_dir"] = str(out_dir)

tmp = tempfile.NamedTemporaryFile(prefix=f"{dataset}_{tag}_", suffix=".yaml", delete=False)
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

  mapfile -t prep < <(prepare_cfg "$BASE_CFG" "$seed" "$split_seed" "$rep_tag" "$DATASET" "$BATCH_SIZE" "$EPOCHS")
  local runtime_cfg="${prep[0]}"
  local run_dir="${prep[1]}"

  echo "[em-resnet50 ${DATASET} fullft ${rep_tag}] seed=$seed split_seed=$split_seed cfg=$runtime_cfg"
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
  echo "[info] SLURM_ARRAY_TASK_ID not set; running $REPEATS reps sequentially"
  for ((rep=0; rep<REPEATS; rep++)); do run_rep "$rep"; done
fi
