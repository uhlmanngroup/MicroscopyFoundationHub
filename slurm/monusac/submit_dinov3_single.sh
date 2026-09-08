#!/usr/bin/env bash
# Submit MoNuSAC DINOv3-L single-dataset jobs (2 tissues × 2 regimes × 5 seeds).
# Usage: bash slurm/monusac/submit_dinov3_single.sh

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

SCRIPT="slurm/monusac/grid_dinov3_large.sbatch"

for DATASET in epithelial lymphocyte; do
  for MODE in head lora; do
    JOB=$(sbatch --array=0-4 \
      --export=ALL,DATASET="$DATASET",TUNING_MODE="$MODE" \
      "$SCRIPT" | awk '{print $NF}')
    echo "submitted  dataset=$DATASET  mode=$MODE  jobid=$JOB"
  done
done
