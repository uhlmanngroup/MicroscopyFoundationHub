#!/usr/bin/bash -l
#SBATCH --job-name=deepbacs-triplet-dinov3l
#SBATCH --time=24:00:00
#SBATCH --gpus=H100:1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%A_%a.out

# -----------------------------------------------------------------------
# DeepBacs triplet (coli + aureus + subtilis) — DINOv3-L/16
#
# Control via env vars:
#   TUNING_MODE  head | lora | fullft    (default: lora)
#   REPEATS      number of seeds to run  (default: 5)
#
# Submit every regime at once:
#   bash slurm/deepbacs/submit_dinov3_triplet.sh
# -----------------------------------------------------------------------

# sbatch runs a spool copy of this file, so resolve the repo from the submit dir.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/lib/common.sh"
dino_peft_init

COMBO="coli-aureus-subtilis"
TUNING_MODE=${TUNING_MODE:-lora}    # head | lora | fullft
BASE_CFG=${BASE_CFG:-configs/cluster/deepbacs/deepbacs_triplet_dinov3_cluster.yaml}
DINO_SIZE=${DINO_SIZE:-vitl16}
BACKBONE_NAME=${BACKBONE_NAME:-dinov3}
REPEATS=${REPEATS:-5}
BASE_SEED=${BASE_SEED:-1}
BASE_SPLIT_SEED=${BASE_SPLIT_SEED:-101}
DATA_ROOT=${DATA_ROOT:-"$(dino_peft_path data_root)/deepbacs/triplet"}

# Full fine-tuning holds the whole backbone in memory; everything else fits at 2.
BATCH_SIZE=${BATCH_SIZE:-}
if [ -z "$BATCH_SIZE" ] && [ "$TUNING_MODE" = "fullft" ]; then
  BATCH_SIZE=1
fi

run_rep() {
  local rep_idx="$1"
  local seed=$((BASE_SEED + rep_idx))
  local split_seed=$((BASE_SPLIT_SEED + rep_idx))
  local tag="rep$((rep_idx + 1))"

  # note: ${extra[@]+...} keeps an empty array safe under `set -u`
  local extra=()
  [ -n "$BATCH_SIZE" ] && extra+=(--set "batch_size=$BATCH_SIZE")
  [ -n "${EPOCHS:-}" ] && extra+=(--set "epochs=$EPOCHS")
  [ -n "${NUM_WORKERS:-}" ] && extra+=(--set "num_workers=$NUM_WORKERS")
  [ -n "${AMP:-}" ] && extra+=(--set "amp=$AMP")

  make_runtime_cfg \
    --base-cfg "$BASE_CFG" \
    --tuning-mode "$TUNING_MODE" \
    --seed "$seed" --split-seed "$split_seed" \
    --task-type "${COMBO}-${TUNING_MODE}" \
    --experiment-id "2026-09-08_deepbacs_triplet_${COMBO}_dinov3_${TUNING_MODE}_seg_${tag}" \
    --prefix "${tag}_" \
    --set "backbone.name=$BACKBONE_NAME" \
    --set "backbone.variant=$DINO_SIZE" \
    --set "dino_size=$DINO_SIZE" \
    --set "train_img_dir=${DATA_ROOT}/train/images" \
    --set "train_mask_dir=${DATA_ROOT}/train/masks" \
    --set "test_img_dir=${DATA_ROOT}/test/images" \
    --set "test_mask_dir=${DATA_ROOT}/test/masks" \
    ${extra[@]+"${extra[@]}"}

  echo "[deepbacs triplet mode=$TUNING_MODE $tag] seed=$seed split_seed=$split_seed"
  train_and_eval "$RUNTIME_CFG" "$RUN_DIR" "$tag"
}

dino_peft_dispatch "$REPEATS" run_rep
