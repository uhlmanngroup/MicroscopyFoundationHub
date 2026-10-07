#!/usr/bin/env bash
# Bash entry point; the shared Python submitter queues the SLURM dependency chain.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source slurm/lib/common.sh
[ -x "$PY" ] || { echo "[error] Python not found: $PY. Export PY=/path/to/dino-peft/bin/python." >&2; exit 1; }
exec "$PY" scripts/joint_metric.py submit "$@"
