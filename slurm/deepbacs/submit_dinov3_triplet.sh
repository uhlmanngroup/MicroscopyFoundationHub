#!/usr/bin/env bash
# Submit DINOv3-L triplet DeepBacs jobs (3 modes × 5 seeds).
# Usage: bash slurm/deepbacs/submit_dinov3_triplet.sh

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

SCRIPT="slurm/deepbacs/grid_triplet_dinov3_large.sbatch"

for MODE in head lora fullft; do
  JOB=$(sbatch --array=0-4 \
    --export=ALL,TUNING_MODE="$MODE" \
    "$SCRIPT" | awk '{print $NF}')
  echo "submitted  triplet  mode=$MODE  jobid=$JOB"
done
