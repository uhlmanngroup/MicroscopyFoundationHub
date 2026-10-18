#!/usr/bin/env bash
# Submit only the five missing corrected joint-EM columns, five seeds each.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

RUN_TAG=${RUN_TAG:-2026-10-18_joint_em_missing}
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "RUN_TAG may contain only letters, numbers, periods, underscores and hyphens" >&2
  exit 1
fi

echo "Preparing and validating TRIPLET_LUCCHI_FIXED in a CPU job..."
sbatch --wait slurm/em/prepare_triplet_lucchi_fixed.sbatch

jobs=()
for CELL in dinov3_lora openclip_lora dinov2_frozen dinov2_lora dinov2_fullft; do
  job_id=$(sbatch --parsable --array=0-4%1 \
    --export=ALL,CELL="$CELL",RUN_TAG="$RUN_TAG" \
    slurm/em/grid_triplet_missing.sbatch)
  job_id=${job_id%%;*}
  jobs+=("$job_id")
  echo "Submitted $CELL: job $job_id, reps 1-5"
done

dependency=$(IFS=:; echo "${jobs[*]}")
summary_job=$(sbatch --parsable --dependency="afterok:$dependency" \
  --export=ALL,RUN_TAG="$RUN_TAG",SUMMARY_CELLS=dinov3_lora,openclip_lora,dinov2_frozen,dinov2_lora,dinov2_fullft \
  slurm/em/summarize_triplet_lucchi_fixed.sbatch)
echo "Submitted five-seed summary: job $summary_job (after all 25 experiments succeed)"
