#!/usr/bin/bash -l
#SBATCH --job-name=feat-domain-pipeline
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# ---- User-overridable settings ----
CFG_PATH="${CFG_PATH:-configs/cluster/EM/triplet_feat_domain_pipeline_dinov2_large_head_center448.yaml}"
CHECKPOINT="${CHECKPOINT:-}"          # optional .pt for LoRA/adapted features
SKIP_FEATURES="${SKIP_FEATURES:-0}"   # 1 to skip extract_features
SKIP_PCA="${SKIP_PCA:-0}"             # 1 to skip PCA/UMAP
SKIP_DOMAIN="${SKIP_DOMAIN:-0}"       # 1 to skip domain analysis
KEEP_TEMP_CFG="${KEEP_TEMP_CFG:-0}"   # 1 to retain generated runtime cfg
DRY_RUN="${DRY_RUN:-0}"               # 1 to print commands only

if [ ! -f "$CFG_PATH" ]; then
  echo "[error] Config file not found: $CFG_PATH" >&2
  exit 2
fi

ARGS=(--cfg "$CFG_PATH")
[ -n "$CHECKPOINT" ] && ARGS+=(--checkpoint "$CHECKPOINT")
[ "$SKIP_FEATURES" = "1" ] && ARGS+=(--skip-features)
[ "$SKIP_PCA" = "1" ] && ARGS+=(--skip-pca)
[ "$SKIP_DOMAIN" = "1" ] && ARGS+=(--skip-domain)
[ "$KEEP_TEMP_CFG" = "1" ] && ARGS+=(--keep-temp-cfg)
[ "$DRY_RUN" = "1" ] && ARGS+=(--dry-run)

echo "[pipeline-sbatch] python=$PY"
echo "[pipeline-sbatch] cfg=$CFG_PATH"
[ -n "$CHECKPOINT" ] && echo "[pipeline-sbatch] checkpoint=$CHECKPOINT"
echo "[pipeline-sbatch] flags: skip_features=$SKIP_FEATURES skip_pca=$SKIP_PCA skip_domain=$SKIP_DOMAIN dry_run=$DRY_RUN"

"$PY" scripts/run_feature_domain_pipeline.py "${ARGS[@]}"
