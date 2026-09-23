#!/usr/bin/bash -l
#SBATCH --job-name=deepbacs-openclip-vitl14
#SBATCH --time=12:00:00
#SBATCH --gpus=H100:1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%A_%a.out

# -----------------------------------------------------------------------
# DeepBacs experiments — OpenCLIP ViT-L-14 (LAION-2B).
#
# Covers the three species and the joint run in one script:
#   DATASET      coli | aureus | subtilis | triplet  (default: subtilis)
#   TUNING_MODE  head | lora | fullft                (default: lora)
#   REPEATS      number of seeds to run              (default: 5)
#
# Seeds match the DINOv3 DeepBacs runs (seed 1-5, split_seed 101-105) so the two
# backbones share a train/val partition per repeat.
#
# DATASET=triplet is the joint E. coli + S. aureus + B. subtilis run. Its task_type
# and experiment_id both keep the substring "triplet", which is what switches on the
# per-species foreground IoU in metrics.json.
#
# Submit everything:
#   bash slurm/deepbacs/submit_deepbacs_openclip.sh
# Or one cell:
#   sbatch --array=0-4 --export=ALL,DATASET=triplet,TUNING_MODE=lora \
#          slurm/deepbacs/grid_deepbacs_openclip_large.sbatch
# -----------------------------------------------------------------------

# sbatch runs a spool copy of this file, so resolve the repo from the submit dir.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/lib/common.sh"
dino_peft_init

DATASET=${DATASET:-subtilis}        # coli | aureus | subtilis | triplet
TUNING_MODE=${TUNING_MODE:-lora}    # head | lora | fullft
OPENCLIP_MODEL=${OPENCLIP_MODEL:-ViT-L-14}
OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-laion2b_s32b_b82k}
BACKBONE_NAME=${BACKBONE_NAME:-openclip}
REPEATS=${REPEATS:-5}
BASE_SEED=${BASE_SEED:-1}
BASE_SPLIT_SEED=${BASE_SPLIT_SEED:-101}
# Send a run somewhere other than the config's results_root — how the pilot in
# slurm/pilot_openclip.sh keeps five-epoch runs out of the real results tree.
RESULTS_ROOT=${RESULTS_ROOT:-}
DATA_ROOT=${DATA_ROOT:-"$(dino_peft_path data_root)/deepbacs"}

case "$DATASET" in
  coli|aureus|subtilis)
    BASE_CFG=${BASE_CFG:-configs/cluster/deepbacs/deepbacs_single_openclip_vitl14_cluster.yaml}
    DATA_BASE="${DATA_ROOT}/single/${DATASET}"
    TASK_TYPE="seg/single/${DATASET}-${TUNING_MODE}"
    EXPERIMENT_BASE="2026-09-23_deepbacs_${DATASET}_openclip_vitl14_${TUNING_MODE}_seg"
    ;;
  triplet)
    BASE_CFG=${BASE_CFG:-configs/cluster/deepbacs/deepbacs_triplet_openclip_vitl14_cluster.yaml}
    DATA_BASE="${DATA_ROOT}/triplet"
    # "triplet" in both of these is load-bearing: it turns on the per-species metrics.
    TASK_TYPE="seg/triplet/coli-aureus-subtilis-${TUNING_MODE}"
    EXPERIMENT_BASE="2026-09-23_deepbacs_triplet_coli-aureus-subtilis_openclip_vitl14_${TUNING_MODE}_seg"
    ;;
  *) echo "[error] unknown DATASET='$DATASET' (coli|aureus|subtilis|triplet)" >&2; exit 1 ;;
esac
case "$TUNING_MODE" in
  head|lora|fullft) ;;
  *) echo "[error] unknown TUNING_MODE='$TUNING_MODE' (head|lora|fullft)" >&2; exit 1 ;;
esac
[ -f "$BASE_CFG" ] || { echo "[error] config not found: $BASE_CFG" >&2; exit 1; }

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
  [ -n "${PATIENCE:-}" ] && extra+=(--set "patience=$PATIENCE")
  [ -n "$RESULTS_ROOT" ] && extra+=(--set "results_root=$RESULTS_ROOT")

  make_runtime_cfg \
    --base-cfg "$BASE_CFG" \
    --tuning-mode "$TUNING_MODE" \
    --seed "$seed" --split-seed "$split_seed" \
    --task-type "$TASK_TYPE" \
    --experiment-id "${EXPERIMENT_BASE}_${tag}" \
    --prefix "${tag}_" \
    --set "backbone.name=$BACKBONE_NAME" \
    --set "backbone.model=$OPENCLIP_MODEL" \
    --set "backbone.variant=$OPENCLIP_MODEL" \
    --set "backbone.pretrained=$OPENCLIP_PRETRAINED" \
    --set "dino_size=$OPENCLIP_MODEL" \
    --set "train_img_dir=${DATA_BASE}/train/images" \
    --set "train_mask_dir=${DATA_BASE}/train/masks" \
    --set "test_img_dir=${DATA_BASE}/test/images" \
    --set "test_mask_dir=${DATA_BASE}/test/masks" \
    ${extra[@]+"${extra[@]}"}

  echo "[deepbacs-openclip $DATASET mode=$TUNING_MODE $tag] seed=$seed split_seed=$split_seed"
  train_and_eval "$RUNTIME_CFG" "$RUN_DIR" "$tag"
}

dino_peft_dispatch "$REPEATS" run_rep
