#!/usr/bin/env bash
# Submit the whole OpenCLIP ViT-L-14 benchmark, then a summary job that waits for it
# and emails the digest.
#
#   EM        lucchi, kasthuri, droso, and the joint triplet   x 2 regimes x 5 seeds
#   DeepBacs  coli, aureus, subtilis, and the joint triplet    x 2 regimes x 5 seeds
#   MoNuSAC   epithelial, lymphocyte                           x 2 regimes x 5 seeds
#   = 20 array jobs, 100 array tasks.
#
# The two regimes are frozen (head-only) and end-to-end. LoRA is out of the sweep by
# default; MODES="head fullft lora" puts it back, and nothing else has to change.
#
# The summary job depends on ALL of them with afterany, not afterok: a sweep where two
# cells crashed should still produce a report saying which ones, rather than no report.
# Cells with fewer than 5 repeats are flagged in the digest.
#
# Usage:
#   bash slurm/submit_openclip_all.sh
#   NOTIFY_EMAIL=you@example.org bash slurm/submit_openclip_all.sh
#   MODES="head fullft lora" bash slurm/submit_openclip_all.sh # add LoRA back
#   DOMAINS="em monusac" bash slurm/submit_openclip_all.sh     # narrow the domains
#   DRY_RUN=1 bash slurm/submit_openclip_all.sh                # print, submit nothing
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

NOTIFY_EMAIL=${NOTIFY_EMAIL:-caterina.fusterbarcelo@mls.uzh.ch}
MODES=${MODES:-"head fullft"}
DOMAINS=${DOMAINS:-"em deepbacs monusac"}
DRY_RUN=${DRY_RUN:-}
SKIP_PREFETCH=${SKIP_PREFETCH:-}
OPENCLIP_MODEL=${OPENCLIP_MODEL:-ViT-L-14}
OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-laion2b_s32b_b82k}
PY=${PY:-$HOME/data/conda/envs/dino-peft/bin/python}
mkdir -p logs

# One download here beats 30 array tasks racing for the same cache entry, and a node
# without network fails now instead of an hour into the sweep. SKIP_PREFETCH=1 opts out.
if [ -z "$SKIP_PREFETCH" ] && [ -z "$DRY_RUN" ]; then
  echo "--- prefetching OpenCLIP weights ---"
  PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" scripts/utils/prefetch_openclip.py \
      --model "$OPENCLIP_MODEL" --pretrained "$OPENCLIP_PRETRAINED"
  echo
fi

declare -a JOB_IDS=()

submit_domain() {
  local script="$1" domain="$2"
  echo "--- $domain ---"
  if [ -n "$DRY_RUN" ]; then
    MODES="$MODES" bash -c "echo '[dry-run] would run: MODES=\"$MODES\" bash $script'"
    return
  fi
  # Each submit_* script prints "... jobid=<N>" per (dataset, regime).
  local out
  out=$(MODES="$MODES" bash "$script")
  echo "$out"
  while read -r id; do
    [ -n "$id" ] && JOB_IDS+=("$id")
  done < <(printf '%s\n' "$out" | sed -n 's/.*jobid=\([0-9][0-9]*\).*/\1/p')
}

for domain in $DOMAINS; do
  case "$domain" in
    em)       submit_domain slurm/em/submit_em_openclip.sh em ;;
    deepbacs) submit_domain slurm/deepbacs/submit_deepbacs_openclip.sh deepbacs ;;
    monusac)  submit_domain slurm/monusac/submit_monusac_openclip.sh monusac ;;
    *) echo "[error] unknown domain '$domain' (em|deepbacs|monusac)" >&2; exit 1 ;;
  esac
done

if [ -n "$DRY_RUN" ]; then
  echo
  echo "[dry-run] would then submit slurm/openclip_summary.sbatch with"
  echo "          --dependency=afterany:<all job ids> --mail-user=$NOTIFY_EMAIL"
  exit 0
fi

if [ ${#JOB_IDS[@]} -eq 0 ]; then
  echo "[error] nothing was submitted; not queueing a summary job." >&2
  exit 1
fi

DEP=$(IFS=:; echo "${JOB_IDS[*]}")
SUMMARY_JOB=$(sbatch \
  --dependency="afterany:${DEP}" \
  --kill-on-invalid-dep=yes \
  --mail-type=END,FAIL \
  --mail-user="$NOTIFY_EMAIL" \
  --export=ALL,NOTIFY_EMAIL="$NOTIFY_EMAIL" \
  slurm/openclip_summary.sbatch | awk '{print $NF}')

echo
echo "submitted  ${#JOB_IDS[@]} sweep jobs"
echo "submitted  summary jobid=$SUMMARY_JOB  (waits for all of them, mails $NOTIFY_EMAIL)"
echo
echo "watch with:  squeue -u \$USER"
echo "the summary lands in <scratch_root>/openclip/summary/{summary.csv,run_metrics.csv,summary.json}"
