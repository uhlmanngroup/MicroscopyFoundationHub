#!/usr/bin/bash -l
#SBATCH --job-name=em-openclip-vitl14
#SBATCH --time=24:00:00
#SBATCH --gpus=H100:1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%A_%a.out

# -----------------------------------------------------------------------
# EM experiments — OpenCLIP ViT-L-14 (LAION-2B).
#
# Covers the three individual sets and the joint run in one script:
#   DATASET      lucchi | kasthuri | droso | triplet  (default: lucchi)
#   TUNING_MODE  head | lora | fullft                 (default: lora)
#   REPEATS      number of seeds to run               (default: 5)
#
# Seeds are deliberately IDENTICAL to the DINOv3 and ResNet50 EM runs (seed 1-5,
# split_seed 101-105) so the backbones are compared on the same train/val partition
# per repeat. Do not change BASE_SEED / BASE_SPLIT_SEED without re-running those too.
#
# DATASET=triplet is the joint VNC + Lucchi++ + Kasthuri++ run. Its task_type and
# experiment_id both keep the substring "triplet", which is what switches on the
# per-source foreground IoU in metrics.json (see the note in the triplet config).
#
# Submit everything:
#   bash slurm/em/submit_em_openclip.sh
# Or one cell of the grid:
#   sbatch --array=0-4 --export=ALL,DATASET=triplet,TUNING_MODE=fullft \
#          slurm/em/grid_em_openclip_large.sbatch
# -----------------------------------------------------------------------

# sbatch runs a spool copy of this file, so resolve the repo from the submit dir.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/lib/common.sh"
dino_peft_init

DATASET=${DATASET:-lucchi}          # lucchi | kasthuri | droso | triplet
TUNING_MODE=${TUNING_MODE:-lora}    # head | lora | fullft
BASE_CFG=${BASE_CFG:-configs/cluster/EM/${DATASET}_openclip_vitl14_cluster.yaml}
OPENCLIP_MODEL=${OPENCLIP_MODEL:-ViT-L-14}
OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-laion2b_s32b_b82k}
BACKBONE_NAME=${BACKBONE_NAME:-openclip}
REPEATS=${REPEATS:-5}
BASE_SEED=${BASE_SEED:-1}
BASE_SPLIT_SEED=${BASE_SPLIT_SEED:-101}
# Send a run somewhere other than the config's results_root — how the pilot in
# slurm/pilot_openclip.sh keeps five-epoch runs out of the real results tree.
RESULTS_ROOT=${RESULTS_ROOT:-}

case "$DATASET" in
  lucchi|kasthuri|droso|triplet) ;;
  *) echo "[error] unknown DATASET='$DATASET' (lucchi|kasthuri|droso|triplet)" >&2; exit 1 ;;
esac
case "$TUNING_MODE" in
  head|lora|fullft) ;;
  *) echo "[error] unknown TUNING_MODE='$TUNING_MODE' (head|lora|fullft)" >&2; exit 1 ;;
esac
[ -f "$BASE_CFG" ] || { echo "[error] config not found: $BASE_CFG" >&2; exit 1; }

# Full fine-tuning holds the whole ViT-L in memory at ~1022 px; the config's own
# batch size is fine for head and LoRA.
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
    --task-type "seg/${DATASET}-${TUNING_MODE}" \
    --experiment-id "2026-09-23_em_${DATASET}_openclip_vitl14_${TUNING_MODE}_seg_${tag}" \
    --prefix "${tag}_" \
    --set "backbone.name=$BACKBONE_NAME" \
    --set "backbone.model=$OPENCLIP_MODEL" \
    --set "backbone.variant=$OPENCLIP_MODEL" \
    --set "backbone.pretrained=$OPENCLIP_PRETRAINED" \
    --set "dino_size=$OPENCLIP_MODEL" \
    ${extra[@]+"${extra[@]}"}

  echo "[em-openclip $DATASET mode=$TUNING_MODE $tag] seed=$seed split_seed=$split_seed"
  train_and_eval "$RUNTIME_CFG" "$RUN_DIR" "$tag"
}

dino_peft_dispatch "$REPEATS" run_rep
