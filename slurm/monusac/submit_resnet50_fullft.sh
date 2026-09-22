#!/usr/bin/env bash
# CNN baseline on MoNuSAC: UNet/ResNet50 end-to-end, 5 seeds per cell type.
#
# Usage:
#   bash slurm/monusac/submit_resnet50_fullft.sh               # both
#   bash slurm/monusac/submit_resnet50_fullft.sh lymphocyte
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

SCRIPT="slurm/monusac/grid_resnet50_fullft.sbatch"
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
  JOB=$(sbatch --array=0-4 --export=ALL,DATASET="$ds" "$SCRIPT" | awk '{print $NF}')
  echo "submitted  monusac-resnet50 ${ds} fullft seeds 1-5  jobid=$JOB"
done
