#!/usr/bin/env bash
# Submit 4 additional EM triplet full fine-tune seeds (rep2–rep5).
# Rep1 already exists from the original experiment.
# Usage: bash slurm/em/submit_em_triplet_fullft.sh

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

SCRIPT="slurm/em/grid_triplet_dinov3_fullft.sbatch"

JOB=$(sbatch --array=0-3 "$SCRIPT" | awk '{print $NF}')
echo "submitted  em-triplet fullft reps 2-5  jobid=$JOB"
