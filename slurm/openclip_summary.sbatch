#!/usr/bin/bash -l
#SBATCH --job-name=openclip-summary
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/%x-%j.out

# -----------------------------------------------------------------------
# Aggregate every OpenCLIP run and mail the digest.
#
# summarize_seg_results.py walks ROOT for metrics.json, so pointing it at the
# backbone root (not one modality) picks up em/, deepbacs/ and monusac/ in one pass;
# each row carries its own modality. Output lands in ROOT/summary/.
#
#   ROOT           results tree to scan (default: <scratch_root>/openclip)
#   NOTIFY_EMAIL   who to mail the digest to
#
# Usually submitted by slurm/submit_openclip_all.sh with a dependency on the sweep,
# but it is safe to run on its own at any time:
#   sbatch --export=ALL,NOTIFY_EMAIL=you@example.org slurm/openclip_summary.sbatch
# -----------------------------------------------------------------------

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/lib/common.sh"
dino_peft_init

ROOT=${ROOT:-"$(dino_peft_path scratch_root)/openclip"}
SUMMARY_DIR_NAME=${SUMMARY_DIR_NAME:-summary}
EXPECTED_REPEATS=${EXPECTED_REPEATS:-5}
NOTIFY_EMAIL=${NOTIFY_EMAIL:-}

echo "[summary] root=$ROOT"
if [ ! -d "$ROOT" ]; then
  echo "[error] results root does not exist: $ROOT" >&2
  exit 1
fi

"$PY" scripts/analysis/summarize_seg_results.py \
  --root "$ROOT" \
  --summary-dir-name "$SUMMARY_DIR_NAME"

MAIL_ARGS=()
[ -n "$NOTIFY_EMAIL" ] && MAIL_ARGS+=(--to "$NOTIFY_EMAIL")

"$PY" scripts/utils/send_summary_email.py \
  --summary-dir "${ROOT}/${SUMMARY_DIR_NAME}" \
  --subject "OpenCLIP ViT-L-14 sweep — results summary" \
  --expected-repeats "$EXPECTED_REPEATS" \
  --note "OpenCLIP ViT-L-14 (laion2b_s32b_b82k), seeds 1-5 / split_seeds 101-105." \
  --note "EM, DeepBacs and MoNuSAC; the joint runs carry per-source foreground IoU." \
  ${MAIL_ARGS[@]+"${MAIL_ARGS[@]}"}

echo "[summary] reports in ${ROOT}/${SUMMARY_DIR_NAME}"
