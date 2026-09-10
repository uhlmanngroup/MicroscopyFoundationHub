#!/usr/bin/env bash
# Rerun EM triplet frozen (head-only) with proper varied seeds.
# Original runs all used seed=0 (split_seed=42), making std=0.
# Usage: bash slurm/em/submit_em_triplet_frozen.sh

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

SCRIPT="slurm/em/grid_triplet_dinov3_frozen.sbatch"

JOB=$(sbatch --array=0-4 "$SCRIPT" | awk '{print $NF}')
echo "submitted  em-triplet frozen reps 1-5  jobid=$JOB"
