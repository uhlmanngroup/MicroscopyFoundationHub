#!/usr/bin/env bash
# Submit all DINOv3-L paired-dataset DeepBacs jobs (3 combos × 3 regimes × 5 seeds).
# Usage: bash slurm/deepbacs/submit_dinov3_paired.sh

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

SCRIPT="slurm/deepbacs/grid_paired_dinov3_large.sbatch"

for COMBO in aureus-subtilis coli-aureus coli-subtilis; do
  for MODE in head lora fullft; do
    JOB=$(sbatch --array=0-4 \
      --export=ALL,COMBO="$COMBO",TUNING_MODE="$MODE" \
      "$SCRIPT" | awk '{print $NF}')
    echo "submitted  combo=$COMBO  mode=$MODE  jobid=$JOB"
  done
done
