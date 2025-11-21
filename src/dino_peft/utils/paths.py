from __future__ import annotations

from datetime import datetime
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


def _require_field(cfg: Mapping[str, Any], key: str) -> Any:
    if key not in cfg or cfg[key] in (None, ""):
        raise ValueError(f"Config must define '{key}'")
    return cfg[key]


def _git_commit() -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return output.decode("utf-8").strip()
    except Exception:
        return "unknown"


def resolve_run_dir(cfg: Mapping[str, Any], task_type: str) -> Path:
    exp_id = _require_field(cfg, "experiment_id")
    results_root = _require_field(cfg, "results_root")
    task = task_type or cfg.get("task_type")
    if not task:
        raise ValueError("task_type must be provided either in cfg or as an argument")
    return Path(results_root).expanduser() / task / exp_id


def setup_run_dir(
    cfg: Mapping[str, Any],
    task_type: str,
    subdirs: Iterable[str] | None = None,
    save_config: bool = True,
) -> Path:
    """
    Create and return the canonical run directory:
    <results_root>/<task_type>/<experiment_id>
    """
    run_dir = resolve_run_dir(cfg, task_type)
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in subdirs or []:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    if save_config:
        with (run_dir / "config_used.yaml").open("w") as f:
            yaml.safe_dump(dict(cfg), f)
    return run_dir


def write_run_info(run_dir: Path, extra: Dict[str, Any] | None = None) -> None:
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
    }
    if extra:
        info.update(extra)
    lines = [f"{key}: {value}\n" for key, value in info.items()]
    (Path(run_dir) / "run_info.txt").write_text("".join(lines))


def update_metrics(
    run_dir: Path,
    section: str,
    metrics: Dict[str, Any],
) -> None:
    metrics_path = Path(run_dir) / "metrics.json"
    existing = {}
    if metrics_path.exists():
        try:
            existing = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            existing = {}
    existing[section] = metrics
    metrics_path.write_text(json.dumps(existing, indent=2, sort_keys=True))
