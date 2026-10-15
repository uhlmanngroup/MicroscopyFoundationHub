#!/usr/bin/bash -l
#SBATCH --job-name=em-triplet-fixed
#SBATCH --time=24:00:00
#SBATCH --gpus=H100:1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%A_%a.out

# One task = one model/regime and one seed. Submit with
# bash slurm/em/submit_triplet_lucchi_fixed.sh after the corrected data is ready.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/lib/common.sh"
dino_peft_init

: "${CELL:?Submit through slurm/em/submit_triplet_lucchi_fixed.sh}"
: "${RUN_TAG:?RUN_TAG must identify this corrected rerun}"

case "$CELL" in
  dinov3_frozen)
    BASE_CFG=configs/cluster/EM/paired_dinov3_cluster.yaml
    TUNING_MODE=head
    TASK_TYPE=seg/triplet-nolora
    BATCH_SIZE=2
    ;;
  dinov3_fullft)
    BASE_CFG=configs/cluster/EM/paired_dinov3_cluster.yaml
    TUNING_MODE=fullft
    TASK_TYPE=seg/triplet-fullft
    BATCH_SIZE=1
    ;;
  openclip_frozen)
    BASE_CFG=configs/cluster/EM/triplet_openclip_vitl14_cluster.yaml
    TUNING_MODE=head
    TASK_TYPE=seg/triplet-head
    BATCH_SIZE=2
    ;;
  openclip_fullft)
    BASE_CFG=configs/cluster/EM/triplet_openclip_vitl14_cluster.yaml
    TUNING_MODE=fullft
    TASK_TYPE=seg/triplet-fullft
    BATCH_SIZE=1
    ;;
  resnet50_fullft)
    BASE_CFG=configs/cluster/EM/triplet_resnet50_fullft_cluster.yaml
    TUNING_MODE=fullft
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
DATASET_ROOT="$(dino_peft_path data_root)/EM/TRIPLET_LUCCHI_FIXED"
LUCCHI_ROOT="$(dino_peft_path data_root)/EM/Lucchi++"

# Fail before reserving a run directory if any Lucchi slice is still mismatched.
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
