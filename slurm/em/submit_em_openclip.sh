#!/usr/bin/env bash
# Submit the OpenCLIP ViT-L-14 EM sweep: 4 datasets x 2 regimes x 5 seeds = 40 array tasks.
#
# Regimes default to frozen + end-to-end. LoRA is deliberately not in the sweep; the
# grid script still accepts TUNING_MODE=lora, so MODES="lora" adds it back.
#
# Usage:
#   bash slurm/em/submit_em_openclip.sh                        # everything
#   bash slurm/em/submit_em_openclip.sh lucchi kasthuri droso  # individual sets only
#   bash slurm/em/submit_em_openclip.sh triplet                # the joint run only
#   MODES="head fullft lora" bash slurm/em/submit_em_openclip.sh   # add LoRA back
#
# Prints one job id per (dataset, regime) so a dependent summary job can wait on them;
# slurm/submit_openclip_all.sh does exactly that.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

SCRIPT="slurm/em/grid_em_openclip_large.sbatch"
MODES=${MODES:-"head fullft"}
mkdir -p logs

DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
  DATASETS=(lucchi kasthuri droso triplet)
fi

for ds in "${DATASETS[@]}"; do
  case "$ds" in
    lucchi|kasthuri|droso|triplet) ;;
    *) echo "[error] unknown dataset '$ds' (lucchi|kasthuri|droso|triplet)" >&2; exit 1 ;;
  esac
done

for ds in "${DATASETS[@]}"; do
  for mode in $MODES; do
    JOB=$(sbatch --array=0-4 --export=ALL,DATASET="$ds",TUNING_MODE="$mode" "$SCRIPT" | awk '{print $NF}')
    echo "submitted  em-openclip  dataset=$ds  mode=$mode  seeds 1-5  jobid=$JOB"
  done
done
