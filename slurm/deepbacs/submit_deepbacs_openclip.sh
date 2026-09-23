#!/usr/bin/env bash
# Submit the OpenCLIP ViT-L-14 DeepBacs sweep: 4 datasets x 2 regimes x 5 seeds
# = 40 array tasks (coli, aureus, subtilis, and the joint triplet).
#
# Regimes default to frozen + end-to-end. LoRA is deliberately not in the sweep; the
# grid script still accepts TUNING_MODE=lora, so MODES="lora" adds it back.
#
# Usage:
#   bash slurm/deepbacs/submit_deepbacs_openclip.sh                    # everything
#   bash slurm/deepbacs/submit_deepbacs_openclip.sh coli aureus subtilis
#   bash slurm/deepbacs/submit_deepbacs_openclip.sh triplet
#   MODES="head fullft lora" bash slurm/deepbacs/submit_deepbacs_openclip.sh
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

SCRIPT="slurm/deepbacs/grid_deepbacs_openclip_large.sbatch"
MODES=${MODES:-"head fullft"}
mkdir -p logs

DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
  DATASETS=(coli aureus subtilis triplet)
fi

for ds in "${DATASETS[@]}"; do
  case "$ds" in
    coli|aureus|subtilis|triplet) ;;
    *) echo "[error] unknown dataset '$ds' (coli|aureus|subtilis|triplet)" >&2; exit 1 ;;
  esac
done

for ds in "${DATASETS[@]}"; do
  for mode in $MODES; do
    JOB=$(sbatch --array=0-4 --export=ALL,DATASET="$ds",TUNING_MODE="$mode" "$SCRIPT" | awk '{print $NF}')
    echo "submitted  deepbacs-openclip  dataset=$ds  mode=$mode  seeds 1-5  jobid=$JOB"
  done
done
