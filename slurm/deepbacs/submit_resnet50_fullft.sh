#!/usr/bin/env bash
# CNN baseline on DeepBacs: UNet/ResNet50 end-to-end, 5 seeds per dataset.
#
# Usage:
#   bash slurm/deepbacs/submit_resnet50_fullft.sh                    # all four
#   bash slurm/deepbacs/submit_resnet50_fullft.sh coli aureus subtilis
#   bash slurm/deepbacs/submit_resnet50_fullft.sh triplet
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

SCRIPT="slurm/deepbacs/grid_resnet50_fullft.sbatch"
mkdir -p logs

DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
  DATASETS=(coli aureus subtilis triplet)
fi

for ds in "${DATASETS[@]}"; do
  case "$ds" in
    coli|aureus|subtilis|triplet) ;;
    *) echo "[error] unknown dataset '$ds' (coli|aureus|subtilis|triplet)" >&2; exit 1 ;;
  esac
done

for ds in "${DATASETS[@]}"; do
  JOB=$(sbatch --array=0-4 --export=ALL,DATASET="$ds" "$SCRIPT" | awk '{print $NF}')
  echo "submitted  deepbacs-resnet50 ${ds} fullft seeds 1-5  jobid=$JOB"
done
