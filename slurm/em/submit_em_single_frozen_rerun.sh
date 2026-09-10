#!/usr/bin/env bash
# Rerun EM single-dataset frozen (head-only) with proper varied seeds.
# Original runs all used seed=0 (std=0). Reruns droso, kasthuri, lucchi.
# Usage: bash slurm/em/submit_em_single_frozen_rerun.sh

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

SCRIPT="slurm/em/grid_em_single_dinov3_frozen_rerun.sbatch"

for DATASET in droso kasthuri lucchi; do
  JOB=$(sbatch --array=0-4 --export=ALL,DATASET="$DATASET" "$SCRIPT" | awk '{print $NF}')
  echo "submitted  em-single frozen $DATASET reps 1-5  jobid=$JOB"
done
