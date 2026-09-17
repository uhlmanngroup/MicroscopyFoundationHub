#!/usr/bin/env bash
# CNN baseline sweep: 4 EM datasets x 5 seeds, UNet/ResNet50 end-to-end.
# Usage: bash slurm/em/submit_em_resnet50_fullft.sh
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

SCRIPT="slurm/em/grid_em_resnet50_fullft.sbatch"

for ds in lucchi kasthuri droso triplet; do
  JOB=$(sbatch --array=0-4 --export=ALL,DATASET="$ds" "$SCRIPT" | awk '{print $NF}')
  echo "submitted  em-resnet50 ${ds} fullft seeds 1-5  jobid=$JOB"
done
