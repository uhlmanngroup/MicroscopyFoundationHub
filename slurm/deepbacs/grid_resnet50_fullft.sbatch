#!/usr/bin/bash -l
#SBATCH --job-name=deepbacs-resnet50-unet-fullft
#SBATCH --time=24:00:00
#SBATCH --gpus=H100:1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%A_%a.out

# -----------------------------------------------------------------------
# CNN baseline on DeepBacs — UNet + ImageNet-supervised ResNet50, end-to-end.
# Control condition for the saliency claim: DeepBacs species share uniform
# saliency, so joint training is expected to stay stable, unlike joint EM.
#
# DATASET: coli | aureus | subtilis | triplet
# Seeds 1-5 / split_seeds 101-105, matching the DINOv3 DeepBacs runs.
#
# Usage:
#   bash slurm/deepbacs/submit_resnet50_fullft.sh
#   sbatch --array=0-4 --export=ALL,DATASET=triplet \
#          slurm/deepbacs/grid_resnet50_fullft.sbatch
# -----------------------------------------------------------------------

set -euo pipefail
# sbatch runs a spool copy of this file, so resolve the repo from the submit dir.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/lib/common.sh"
dino_peft_init

DATASET=${DATASET:-subtilis}        # coli | aureus | subtilis | triplet
REPEATS=${REPEATS:-5}
BASE_SEED=${BASE_SEED:-1}
BASE_SPLIT_SEED=${BASE_SPLIT_SEED:-101}

# BatchNorm needs a batch. At 448x448 a 32M UNet fits many samples per step, and
# batch 1 across heterogeneous sources thrashes the running statistics — that is
# what broke the EM joint runs. Keep this identical for single and joint, or the
# individual-vs-joint comparison is confounded by batch size.
BATCH_SIZE=${BATCH_SIZE:-8}

COMBO="coli-aureus-subtilis"
if [ "$DATASET" = "triplet" ]; then
  BASE_CFG=${BASE_CFG:-configs/cluster/deepbacs/deepbacs_triplet_resnet50_cluster.yaml}
  DATA_BASE=${DATA_ROOT:-"$(dino_peft_path data_root)/deepbacs/triplet"}
  TASK_TYPE="seg/triplet/${COMBO}-fullft"
  EXP_STEM="2026-09-22_deepbacs_triplet_${COMBO}_resnet50-unet_fullft_seg"
else
  case "$DATASET" in
    coli|aureus|subtilis) ;;
    *) echo "[error] unknown DATASET='$DATASET' (coli|aureus|subtilis|triplet)" >&2; exit 1 ;;
  esac
  BASE_CFG=${BASE_CFG:-configs/cluster/deepbacs/deepbacs_single_resnet50_cluster.yaml}
  DATA_BASE=${DATA_ROOT:-"$(dino_peft_path data_root)/deepbacs/single/${DATASET}"}
  TASK_TYPE="seg/single/${DATASET}-fullft"
  EXP_STEM="2026-09-22_deepbacs_${DATASET}_resnet50-unet_fullft_seg"
fi
[ -f "$BASE_CFG" ] || { echo "[error] config not found: $BASE_CFG" >&2; exit 1; }

run_rep() {
  local rep_idx="$1"
  local seed=$((BASE_SEED + rep_idx))
  local split_seed=$((BASE_SPLIT_SEED + rep_idx))
  local tag="rep$((rep_idx + 1))"

  # note: ${extra[@]+...} keeps an empty array safe under `set -u`
  local extra=()
  [ -n "$BATCH_SIZE" ]        && extra+=(--set "batch_size=$BATCH_SIZE")
  [ -n "${EPOCHS:-}" ]        && extra+=(--set "epochs=$EPOCHS")
  [ -n "${NUM_WORKERS:-}" ]   && extra+=(--set "num_workers=$NUM_WORKERS")
  [ -n "${AMP:-}" ]           && extra+=(--set "amp=$AMP")

  make_runtime_cfg \
    --base-cfg "$BASE_CFG" \
    --tuning-mode fullft \
    --seed "$seed" --split-seed "$split_seed" \
    --task-type "$TASK_TYPE" \
    --experiment-id "${EXP_STEM}_${tag}" \
    --prefix "${tag}_" \
    --set "backbone.name=resnet50" \
    --set "backbone.variant=resnet50" \
    --set "backbone.pretrained=imagenet" \
    --set "train_img_dir=${DATA_BASE}/train/images" \
    --set "train_mask_dir=${DATA_BASE}/train/masks" \
    --set "test_img_dir=${DATA_BASE}/test/images" \
    --set "test_mask_dir=${DATA_BASE}/test/masks" \
    ${extra[@]+"${extra[@]}"}

  echo "[deepbacs resnet50 ${DATASET} fullft ${tag}] seed=$seed split_seed=$split_seed batch=$BATCH_SIZE"
  train_and_eval "$RUNTIME_CFG" "$RUN_DIR" "$tag"
}

dino_peft_dispatch "$REPEATS" run_rep
