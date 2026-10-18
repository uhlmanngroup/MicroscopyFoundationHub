#!/usr/bin/bash -l
#SBATCH --job-name=em-triplet-missing
#SBATCH --time=24:00:00
#SBATCH --gpus=H100:1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%A_%a.out

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/lib/common.sh"
dino_peft_init

: "${CELL:?Submit through slurm/em/submit_triplet_missing.sh}"
: "${RUN_TAG:?RUN_TAG must identify this rerun}"

case "$CELL" in
  dinov3_lora)
    BASE_CFG=configs/cluster/EM/paired_dinov3_cluster.yaml
    TASK_TYPE=seg/triplet-lora
    BATCH_SIZE=2
    ;;
  openclip_lora)
    BASE_CFG=configs/cluster/EM/triplet_openclip_vitl14_cluster.yaml
    TASK_TYPE=seg/triplet-lora
    BATCH_SIZE=2
    ;;
  dinov2_frozen)
    BASE_CFG=configs/cluster/EM/triplet_dinov2_large_cluster.yaml
    TASK_TYPE=seg/triplet-head
    BATCH_SIZE=2
    ;;
  dinov2_lora)
    BASE_CFG=configs/cluster/EM/triplet_dinov2_large_cluster.yaml
    TASK_TYPE=seg/triplet-lora
    BATCH_SIZE=2
    ;;
  dinov2_fullft)
    BASE_CFG=configs/cluster/EM/triplet_dinov2_large_cluster.yaml
    TASK_TYPE=seg/triplet-fullft
    BATCH_SIZE=1
    ;;
  *) echo "Unknown CELL=$CELL" >&2; exit 1 ;;
esac

rep_idx=${SLURM_ARRAY_TASK_ID:?Submit as --array=0-4}
if (( rep_idx < 0 || rep_idx > 4 )); then
  echo "Array index must be 0..4; got $rep_idx" >&2
  exit 1
fi
rep=$((rep_idx + 1))
seed=$rep
split_seed=$((100 + rep))
tag="rep$rep"
case "$CELL" in
  *_frozen) TUNING_MODE=head ;;
  *_lora) TUNING_MODE=lora ;;
  *_fullft) TUNING_MODE=fullft ;;
esac

DATASET_ROOT="$(dino_peft_path data_root)/EM/TRIPLET_LUCCHI_FIXED"
LUCCHI_ROOT="$(dino_peft_path data_root)/EM/Lucchi++"
"$PY" scripts/data/validate_triplet_lucchi_fixed.py \
  --dataset-root "$DATASET_ROOT" --lucchi-root "$LUCCHI_ROOT"

make_runtime_cfg \
  --base-cfg "$BASE_CFG" \
  --tuning-mode "$TUNING_MODE" \
  --seed "$seed" --split-seed "$split_seed" \
  --task-type "$TASK_TYPE" \
  --experiment-id "${RUN_TAG}_em_triplet_${CELL}_seg_${tag}" \
  --prefix "${CELL}_${tag}_" \
  --set "batch_size=$BATCH_SIZE" \
  --set "epochs=1000" \
  --set "lora_rank=16" --set "lora_alpha=32" \
  --set "results_root=$(dino_peft_path scratch_root)/${CELL%%_*}" \
  --set "train_img_dir=$DATASET_ROOT/train/images" \
  --set "train_mask_dir=$DATASET_ROOT/train/masks" \
  --set "test_img_dir=$DATASET_ROOT/test/images" \
  --set "test_mask_dir=$DATASET_ROOT/test/masks"

if [ -e "$RUN_DIR" ]; then
  echo "Run directory already exists; refusing to overwrite: $RUN_DIR" >&2
  exit 1
fi

echo "[$CELL $tag] seed=$seed split_seed=$split_seed data=$DATASET_ROOT"
echo "[$CELL $tag] config=$RUNTIME_CFG output=$RUN_DIR"
train_and_eval "$RUNTIME_CFG" "$RUN_DIR" "$tag"
