#!/usr/bin/env bash
# Five-epoch pilot of the OpenCLIP sweep — run this BEFORE submit_openclip_all.sh.
#
# Six jobs, one seed each, chosen to touch everything the full sweep touches:
#
#   em triplet      head + fullft   heaviest input (5329 tokens at 1022 px) and the
#                                   per-source foreground IoU path
#   deepbacs triplet head + fullft  native 448 crop, per-species metrics
#   monusac epithelial head + fullft the third modality and its own data tree
#
# What it is for: every run now records peak GPU memory and seconds per epoch in
# metrics.json, so this answers "does end-to-end fit, and how long will 100 tasks take"
# from measurements instead of an estimate. Read the answer with:
#
#   bash slurm/pilot_openclip.sh --report
#
# Results go to <scratch_root>/openclip-pilot, NOT the real tree, so the pilot can
# never contaminate the sweep or its summary. Delete that directory when done.
#
# Usage:
#   bash slurm/pilot_openclip.sh            # submit the pilot
#   bash slurm/pilot_openclip.sh --report   # read the results once it has run
#   EPOCHS=10 bash slurm/pilot_openclip.sh  # longer pilot
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

EPOCHS=${EPOCHS:-5}
# Without this, patience (20) never triggers in 5 epochs — harmless, but it also means
# early stopping is never exercised. Keep it below EPOCHS only if you want to test that.
PATIENCE=${PATIENCE:-20}
PY=${PY:-$HOME/data/conda/envs/dino-peft/bin/python}
PILOT_ROOT=${PILOT_ROOT:-}
PILOT_TIME=${PILOT_TIME:-02:00:00}

resolve_pilot_root() {
  if [ -z "$PILOT_ROOT" ]; then
    PILOT_ROOT="$(PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" -c "
from dino_peft.config import load_paths
print(load_paths()['scratch_root'])")/openclip-pilot"
  fi
}

if [ "${1:-}" = "--report" ]; then
  resolve_pilot_root
  PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" - "$PILOT_ROOT" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
paths = sorted(root.rglob("metrics.json"))
if not paths:
    raise SystemExit(
        f"no finished pilot runs under {root} — still queued, or all failed.\n"
        f"Check `squeue -u $USER` and logs/openclip-pilot-*.out"
    )

runs = [(p, json.loads(p.read_text())) for p in paths]

print(f"{'modality':<9} {'dataset / regime':<34} {'ep':>3} {'s/epoch':>8} {'peak GPU':>9} {'fg IoU':>8}  per-source")
print("-" * 112)
for path, m in runs:
    train, ev = m.get("train", {}), m.get("eval", {})
    per_source = ev.get("foreground_iou_per_dataset") or {}
    spe = train.get("train_seconds_per_epoch")
    gib = train.get("peak_gpu_mem_gib")
    iou = ev.get("foreground_iou")
    # run dir is <root>/<modality>/seg/[<family>/]<group>/<experiment_id>; the group
    # carries dataset and regime, which is what identifies the cell at a glance.
    print(
        f"{train.get('modality', '?'):<9} "
        f"{path.parent.parent.name:<34} "
        f"{train.get('last_epoch', '?'):>3} "
        f"{(f'{spe:.1f}' if spe else '--'):>8} "
        f"{(f'{gib:.1f}GiB' if gib else '--'):>9} "
        f"{(f'{iou:.4f}' if isinstance(iou, (int, float)) else '--'):>8}  "
        f"{', '.join(sorted(per_source)) if per_source else '-'}"
    )

peaks = [m["train"]["peak_gpu_mem_gib"] for _, m in runs if m.get("train", {}).get("peak_gpu_mem_gib")]
rates = [m["train"]["train_seconds_per_epoch"] for _, m in runs if m.get("train", {}).get("train_seconds_per_epoch")]
print()
if peaks:
    print(f"peak GPU across pilot runs : {max(peaks):.1f} GiB   (an H100 has 80 GiB)")
if rates:
    slowest = max(rates)
    print(f"slowest epoch              : {slowest:.1f} s   -> a 24 h job fits ~{int(86400 / slowest)} epochs")
print()
print("Both joint runs (em/deepbacs triplet) must list three names under 'per-source'.")
print("If one shows '-', the per-dataset foreground IoU is missing and the sweep is not ready.")
PY
  exit 0
fi

resolve_pilot_root
echo "pilot results root: $PILOT_ROOT"
echo "epochs=$EPOCHS  patience=$PATIENCE"
echo

submit() {
  local script="$1" dataset="$2" mode="$3"
  local job
  # A short --time schedules sooner than the grid script's own 24 h header, which
  # matters when the whole point is a fast answer.
  job=$(sbatch --array=0-0 \
    --job-name="openclip-pilot" \
    --time="$PILOT_TIME" \
    --export=ALL,DATASET="$dataset",TUNING_MODE="$mode",EPOCHS="$EPOCHS",PATIENCE="$PATIENCE",RESULTS_ROOT="$PILOT_ROOT" \
    "$script" | awk '{print $NF}')
  echo "submitted  pilot  $(basename "$script" .sbatch)  dataset=$dataset  mode=$mode  jobid=$job"
}

for mode in head fullft; do
  submit slurm/em/grid_em_openclip_large.sbatch             triplet    "$mode"
  submit slurm/deepbacs/grid_deepbacs_openclip_large.sbatch triplet    "$mode"
  submit slurm/monusac/grid_monusac_openclip_large.sbatch   epithelial "$mode"
done

echo
echo "when they finish:  bash slurm/pilot_openclip.sh --report"
