#!/usr/bin/env bash
# Submit inference-only transfer evaluation, then summarize on full success.
# Usage: bash slurm/submit_transferability_dinov3_frozen.sh

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

INFER_JOB=$(sbatch --array=0-29 slurm/transferability_dinov3_frozen.sbatch | awk '{print $NF}')
SUMMARY_JOB=$(sbatch --dependency="afterok:${INFER_JOB}" slurm/transferability_aggregate.sbatch | awk '{print $NF}')
echo "submitted inference array jobid=$INFER_JOB (30 tasks; 60 directed evaluations)"
echo "submitted summary jobid=$SUMMARY_JOB (runs after all inference tasks succeed)"
