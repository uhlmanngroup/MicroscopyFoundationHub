#!/usr/bin/bash -l
#SBATCH --job-name=missing-microscopy
#SBATCH --time=24:00:00
#SBATCH --gpus=H100:1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%A_%a.out

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/lib/common.sh"
dino_peft_init

: "${CELL:?Submit through slurm/missing_20261020/submit_missing.sh}"
: "${RUN_TAG:?RUN_TAG is required}"
rep_idx=${SLURM_ARRAY_TASK_ID:?Submit as --array=0-4}
if (( rep_idx < 0 || rep_idx > 4 )); then
  echo "Array index must be 0..4" >&2
  exit 1
fi
rep=$((rep_idx + 1))
tag="rep$rep"
seed=$rep
split_seed=$((100 + rep))
data_root="$(dino_peft_path data_root)"
scratch_root="$(dino_peft_path scratch_root)"

case "$CELL" in
  dinov2_triplet_head|dinov2_triplet_lora|dinov2_triplet_fullft)
    BASE_CFG=configs/cluster/deepbacs/deepbacs_triplet_dinov2_cluster.yaml
    MODE=${CELL##*_}
    [ "$MODE" = head ] && TASK_SUFFIX=head || TASK_SUFFIX="$MODE"
    TASK_TYPE="seg/triplet/coli-aureus-subtilis-${TASK_SUFFIX}"
    FAMILY=dinov2
    MODALITY=deepbacs
    DATA_BASE="$data_root/deepbacs/triplet"
    BATCH_SIZE=2
    [ "$MODE" = fullft ] && BATCH_SIZE=1
    extra=(--set backbone.name=dinov2 --set backbone.variant=large
           --set dino_size=large --set lora_rank=8 --set lora_alpha=16)
    ;;
  openclip_aureus_lora|openclip_coli_lora|openclip_subtilis_lora|openclip_triplet_lora)
    DATASET=${CELL#openclip_}
    DATASET=${DATASET%_lora}
    MODE=lora
    FAMILY=openclip
    MODALITY=deepbacs
    BATCH_SIZE=2
    if [ "$DATASET" = triplet ]; then
      BASE_CFG=configs/cluster/deepbacs/deepbacs_triplet_openclip_vitl14_cluster.yaml
      DATA_BASE="$data_root/deepbacs/triplet"
      TASK_TYPE=seg/triplet/coli-aureus-subtilis-lora
    else
      BASE_CFG=configs/cluster/deepbacs/deepbacs_single_openclip_vitl14_cluster.yaml
      DATA_BASE="$data_root/deepbacs/single/$DATASET"
      TASK_TYPE="seg/single/${DATASET}-lora"
    fi
    extra=(--set backbone.name=openclip --set backbone.model=ViT-L-14
           --set backbone.variant=ViT-L-14
           --set backbone.pretrained=laion2b_s32b_b82k
           --set dino_size=ViT-L-14 --set lora_rank=16 --set lora_alpha=32)
    ;;
  dinov2_vnc_lora)
    BASE_CFG=configs/cluster/EM/droso_dinov2_lora_rerun_cluster.yaml
    MODE=lora
    FAMILY=dinov2
    MODALITY=em
    BATCH_SIZE=1
    TASK_TYPE=seg/single/droso-lora
    DATA_BASE="$data_root/EM/Droso"
    extra=()
    ;;
  *) echo "Unknown CELL=$CELL" >&2; exit 1 ;;
esac

for split in train test; do
  if [ "$CELL" = dinov2_vnc_lora ]; then
    [ "$split" = train ] && split_dir=Train || split_dir=Test
    img_dir="$DATA_BASE/${split_dir}_In"
    mask_dir="$DATA_BASE/${split_dir}_Out"
  else
    img_dir="$DATA_BASE/$split/images"
    mask_dir="$DATA_BASE/$split/masks"
  fi
  [ -d "$img_dir" ] && [ -d "$mask_dir" ] || {
    echo "Missing dataset directory: $img_dir or $mask_dir" >&2
    exit 1
  }
  if [ "$split" = train ]; then
    train_img_dir=$img_dir
    train_mask_dir=$mask_dir
  else
    test_img_dir=$img_dir
    test_mask_dir=$mask_dir
  fi
done

make_runtime_cfg \
  --base-cfg "$BASE_CFG" \
  --tuning-mode "$MODE" \
  --seed "$seed" --split-seed "$split_seed" \
  --task-type "$TASK_TYPE" \
  --experiment-id "${RUN_TAG}_${CELL}_seg_${tag}" \
  --prefix "${CELL}_${tag}_" \
  --set "batch_size=$BATCH_SIZE" \
  --set "epochs=1000" \
  --set "results_root=$scratch_root/$FAMILY" \
  --set "train_img_dir=$train_img_dir" \
  --set "train_mask_dir=$train_mask_dir" \
  --set "test_img_dir=$test_img_dir" \
  --set "test_mask_dir=$test_mask_dir" \
  "${extra[@]}"

if [ -e "$RUN_DIR" ]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  exit 1
fi

echo "[$CELL $tag] seed=$seed split_seed=$split_seed output=$RUN_DIR"
train_and_eval "$RUNTIME_CFG" "$RUN_DIR" "$tag"
