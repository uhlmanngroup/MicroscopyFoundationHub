#!/usr/bin/env bash
# CNN baseline sweep: UNet/ResNet50 end-to-end, 5 seeds per dataset.
#
# Usage:
#   bash slurm/em/submit_em_resnet50_fullft.sh                       # all four
#   bash slurm/em/submit_em_resnet50_fullft.sh lucchi kasthuri droso # individuals only
#   bash slurm/em/submit_em_resnet50_fullft.sh triplet               # joint run only
#
# Seeds default to 1-5 / split_seeds 101-105, matching the DINOv3 EM runs so the
# comparison is paired. Override with BASE_SEED= / BASE_SPLIT_SEED= if they differ.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

SCRIPT="slurm/em/grid_em_resnet50_fullft.sbatch"
mkdir -p logs

DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
  DATASETS=(lucchi kasthuri droso triplet)
fi

for ds in "${DATASETS[@]}"; do
  case "$ds" in
    lucchi|kasthuri|droso|triplet) ;;
    *) echo "[error] unknown dataset '$ds' (lucchi|kasthuri|droso|triplet)" >&2; exit 1 ;;
  esac
done

for ds in "${DATASETS[@]}"; do
  JOB=$(sbatch --array=0-4 --export=ALL,DATASET="$ds" "$SCRIPT" | awk '{print $NF}')
  echo "submitted  em-resnet50 ${ds} fullft seeds 1-5  jobid=$JOB"
done
