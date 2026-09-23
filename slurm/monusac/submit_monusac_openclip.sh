#!/usr/bin/env bash
# Submit the OpenCLIP ViT-L-14 MoNuSAC sweep: 2 cell types x 2 regimes x 5 seeds
# = 20 array tasks.
#
# Regimes default to frozen + end-to-end. LoRA is deliberately not in the sweep; the
# grid script still accepts TUNING_MODE=lora, so MODES="lora" adds it back.
#
# Usage:
#   bash slurm/monusac/submit_monusac_openclip.sh                 # everything
#   bash slurm/monusac/submit_monusac_openclip.sh epithelial      # one cell type
#   MODES="head fullft lora" bash slurm/monusac/submit_monusac_openclip.sh  # add LoRA
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

SCRIPT="slurm/monusac/grid_monusac_openclip_large.sbatch"
MODES=${MODES:-"head fullft"}
mkdir -p logs

DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
  DATASETS=(epithelial lymphocyte)
fi

for ds in "${DATASETS[@]}"; do
  case "$ds" in
    epithelial|lymphocyte) ;;
    *) echo "[error] unknown dataset '$ds' (epithelial|lymphocyte)" >&2; exit 1 ;;
  esac
done

for ds in "${DATASETS[@]}"; do
  for mode in $MODES; do
    JOB=$(sbatch --array=0-4 --export=ALL,DATASET="$ds",TUNING_MODE="$mode" "$SCRIPT" | awk '{print $NF}')
    echo "submitted  monusac-openclip  dataset=$ds  mode=$mode  seeds 1-5  jobid=$JOB"
  done
done
