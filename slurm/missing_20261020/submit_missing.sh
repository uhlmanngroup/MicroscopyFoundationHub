#!/usr/bin/env bash
# Submit the seven absent DeepBacs cells and DINOv2 VNC LoRA, five seeds each.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
RUN_TAG=${RUN_TAG:-2026-10-20_missing_paper}
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "RUN_TAG contains unsupported characters" >&2
  exit 1
fi
cells=(dinov2_triplet_head dinov2_triplet_lora dinov2_triplet_fullft
       openclip_aureus_lora openclip_coli_lora openclip_subtilis_lora
       openclip_triplet_lora dinov2_vnc_lora)
jobs=()
for cell in "${cells[@]}"; do
  job=$(sbatch --parsable --array=0-4%1 \
    --export=ALL,CELL="$cell",RUN_TAG="$RUN_TAG" \
    slurm/missing_20261020/grid_missing.sbatch)
  job=${job%%;*}
  jobs+=("$job")
  echo "Submitted $cell seeds 1-5: $job"
done
dependency=$(IFS=:; echo "${jobs[*]}")
summary=$(sbatch --parsable --dependency="afterany:$dependency" \
  --export=ALL,RUN_TAG="$RUN_TAG" \
  slurm/missing_20261020/summarize_missing.sbatch)
echo "Submitted summary: $summary"
echo "Results: $(sed -n 's/^scratch_root: //p' configs/paths.yaml)/summaries/$RUN_TAG/foreground_iou_mean_std.csv"
