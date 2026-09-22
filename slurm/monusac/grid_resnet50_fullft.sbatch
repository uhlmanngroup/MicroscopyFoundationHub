#!/usr/bin/bash -l
#SBATCH --job-name=monusac-resnet50-unet-fullft
#SBATCH --time=24:00:00
#SBATCH --gpus=H100:1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%A_%a.out

# -----------------------------------------------------------------------
# CNN baseline on MoNuSAC — UNet + ImageNet-supervised ResNet50, end-to-end.
#
# DATASET: epithelial | lymphocyte
# Seeds 1-5 / split_seeds 101-105, matching the DINOv3 MoNuSAC runs.
#
# There is no composed joint MoNuSAC dataset, so this covers the individual
# cell types only — no joint-training collapse test for this modality.
#
# Usage:
#   bash slurm/monusac/submit_resnet50_fullft.sh
#   sbatch --array=0-4 --export=ALL,DATASET=lymphocyte \
#          slurm/monusac/grid_resnet50_fullft.sbatch
# -----------------------------------------------------------------------

set -euo pipefail
# sbatch runs a spool copy of this file, so resolve the repo from the submit dir.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/lib/common.sh"
dino_peft_init

DATASET=${DATASET:-epithelial}      # epithelial | lymphocyte
case "$DATASET" in
  epithelial|lymphocyte) ;;
  *) echo "[error] unknown DATASET='$DATASET' (epithelial|lymphocyte)" >&2; exit 1 ;;
esac

BASE_CFG=${BASE_CFG:-configs/cluster/monusac/monusac_${DATASET}_resnet50_cluster.yaml}
REPEATS=${REPEATS:-5}
BASE_SEED=${BASE_SEED:-1}
BASE_SPLIT_SEED=${BASE_SPLIT_SEED:-101}
DATA_ROOT=${DATA_ROOT:-"$(dino_peft_path data_root)/monusac_448"}

# BatchNorm needs a batch; at 448x448 a 32M UNet fits many samples per step.
BATCH_SIZE=${BATCH_SIZE:-8}

[ -f "$BASE_CFG" ] || { echo "[error] config not found: $BASE_CFG" >&2; exit 1; }

run_rep() {
  local rep_idx="$1"
  local seed=$((BASE_SEED + rep_idx))
  local split_seed=$((BASE_SPLIT_SEED + rep_idx))
  local tag="rep$((rep_idx + 1))"
  local data_base="${DATA_ROOT}/${DATASET}"

  # note: ${extra[@]+...} keeps an empty array safe under `set -u`
  local extra=()
  [ -n "$BATCH_SIZE" ]      && extra+=(--set "batch_size=$BATCH_SIZE")
  [ -n "${EPOCHS:-}" ]      && extra+=(--set "epochs=$EPOCHS")
  [ -n "${NUM_WORKERS:-}" ] && extra+=(--set "num_workers=$NUM_WORKERS")
  [ -n "${AMP:-}" ]         && extra+=(--set "amp=$AMP")

  make_runtime_cfg \
    --base-cfg "$BASE_CFG" \
    --tuning-mode fullft \
    --seed "$seed" --split-seed "$split_seed" \
    --task-type "seg/${DATASET}-fullft" \
    --experiment-id "2026-09-22_monusac_${DATASET}_resnet50-unet_fullft_seg_${tag}" \
    --prefix "${tag}_" \
    --set "backbone.name=resnet50" \
    --set "backbone.variant=resnet50" \
    --set "backbone.pretrained=imagenet" \
    --set "train_img_dir=${data_base}/train/images" \
    --set "train_mask_dir=${data_base}/train/masks" \
    --set "test_img_dir=${data_base}/test/images" \
    --set "test_mask_dir=${data_base}/test/masks" \
    ${extra[@]+"${extra[@]}"}

  echo "[monusac resnet50 ${DATASET} fullft ${tag}] seed=$seed split_seed=$split_seed batch=$BATCH_SIZE"
  train_and_eval "$RUNTIME_CFG" "$RUN_DIR" "$tag"
}

dino_peft_dispatch "$REPEATS" run_rep
