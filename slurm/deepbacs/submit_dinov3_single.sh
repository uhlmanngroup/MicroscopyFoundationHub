#!/usr/bin/env bash
# Submit all DINOv3-L single-dataset DeepBacs jobs (3 species × 3 regimes × 5 seeds).
# Usage: bash slurm/deepbacs/submit_dinov3_single.sh

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

SCRIPT="slurm/deepbacs/grid_dinov3_large.sbatch"

for DATASET in coli aureus subtilis; do
  for MODE in head lora fullft; do
    JOB=$(sbatch --array=0-4 \
      --export=ALL,DATASET="$DATASET",TUNING_MODE="$MODE" \
      "$SCRIPT" | awk '{print $NF}')
    echo "submitted  dataset=$DATASET  mode=$MODE  jobid=$JOB"
  done
done
