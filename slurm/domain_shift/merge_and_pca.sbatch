#!/usr/bin/bash -l
#SBATCH --job-name=domain-shift-pca
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module --quiet load miniforge3 || module load miniforge3

PY="${PY:-$HOME/data/conda/envs/dino-peft/bin/python}"
[ -x "$PY" ] || { echo "[error] Python not found at $PY" >&2; exit 1; }

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TQDM_DISABLE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

PCA_CFG_PATH="${PCA_CFG_PATH:-configs/cluster/domain_shift/domain_shift_3way_pca_dinov2_large.yaml}"
MERGED_OUTPUT="${MERGED_OUTPUT:-/home/cfuste/data/DINO-LoRA/dinov2/domain-analysis/3way/2026-03-30_domain_shift_3way_dinov2-large/features_merged.npz}"
EM_NPZ="${EM_NPZ:-/home/cfuste/data/DINO-LoRA/dinov2/em/feat-analysis/domain-shift/2026-03-30_domain_shift_em_dinov2-large/features.npz}"
DEEPBACS_NPZ="${DEEPBACS_NPZ:-/home/cfuste/data/DINO-LoRA/dinov2/deepbacs/feat-analysis/domain-shift/2026-03-30_domain_shift_deepbacs_dinov2-large/features.npz}"
OPENIMAGES_NPZ="${OPENIMAGES_NPZ:-/home/cfuste/data/DINO-LoRA/dinov2/openimages/feat-analysis/domain-shift/2026-03-30_domain_shift_openimages_dinov2-large/features.npz}"

if [ ! -f "$PCA_CFG_PATH" ]; then
  echo "[error] PCA config file not found: $PCA_CFG_PATH" >&2
  exit 2
fi

echo "[merge-pca-sbatch] python=$PY"
echo "[merge-pca-sbatch] merged_output=$MERGED_OUTPUT"
echo "[merge-pca-sbatch] pca_cfg=$PCA_CFG_PATH"

"$PY" scripts/analysis/merge_feature_npzs.py \
  --output "$MERGED_OUTPUT" \
  --input "em=$EM_NPZ" \
  --input "deepbacs=$DEEPBACS_NPZ" \
  --input "openimages=$OPENIMAGES_NPZ"

"$PY" scripts/run_pca.py --cfg "$PCA_CFG_PATH"
