#!/usr/bin/env bash
# Prepare/validate the corrected triplet, then submit all five table columns.
# Each column runs seeds 1..5 and split seeds 101..105.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

RUN_TAG=${RUN_TAG:-2026-10-15_lucchi_fixed}
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "RUN_TAG may contain only letters, numbers, periods, underscores and hyphens" >&2
  exit 1
fi

echo "Preparing and validating TRIPLET_LUCCHI_FIXED in a CPU job..."
sbatch --wait slurm/em/prepare_triplet_lucchi_fixed.sbatch

jobs=()
for CELL in dinov3_frozen dinov3_fullft openclip_frozen openclip_fullft resnet50_fullft; do
  job_id=$(sbatch --parsable --array=0-4%1 \
    --export=ALL,CELL="$CELL",RUN_TAG="$RUN_TAG" \
    slurm/em/grid_triplet_lucchi_fixed.sbatch)
  job_id=${job_id%%;*}
  jobs+=("$job_id")
  echo "Submitted $CELL: job $job_id, reps 1-5"
done

dependency=$(IFS=:; echo "${jobs[*]}")
summary_job=$(sbatch --parsable --dependency="afterok:$dependency" \
  --export=ALL,RUN_TAG="$RUN_TAG" \
  slurm/em/summarize_triplet_lucchi_fixed.sbatch)
echo "Submitted five-seed summary: job $summary_job (runs after all 25 experiments succeed)"
