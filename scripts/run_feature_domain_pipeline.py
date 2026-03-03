#!/usr/bin/env python3
"""Run feature extraction -> PCA/UMAP -> domain analysis with one config.

This wrapper reuses the existing scripts and injects the derived `features.npz` path
into the PCA and domain-analysis stages so you only maintain one config file.

Typical usage:
    python scripts/run_feature_domain_pipeline.py --cfg configs/cluster/paired_feat_analysis_cluster.yaml

Notes:
  - The config should be primarily an `extract_features.py` config (data/model/runtime).
  - Add `pca:` / `umap:` settings for PCA/UMAP.
  - Add `domain_analysis:` settings for FDD/LR. If no explicit binary domains are
    provided, this wrapper defaults to `pairwise_all` using the extracted features.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dino_peft.utils.paths import resolve_run_dir

DEFAULT_CFG = REPO_ROOT / "configs" / "mac" / "em_unsupervised_features_mac.yaml"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run extract_features + PCA/UMAP + domain analysis in one command."
    )
    ap.add_argument(
        "--cfg",
        "--config",
        dest="cfg",
        type=str,
        default=str(DEFAULT_CFG),
        help="Path to unified YAML config.",
    )
    ap.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint (.pt) passed through to extract_features.py",
    )
    ap.add_argument("--skip-features", action="store_true", help="Skip feature extraction stage")
    ap.add_argument("--skip-pca", action="store_true", help="Skip PCA/UMAP stage")
    ap.add_argument("--skip-domain", action="store_true", help="Skip domain analysis stage")
    ap.add_argument(
        "--keep-temp-cfg",
        action="store_true",
        help="Keep the generated runtime config for debugging",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands and paths without executing",
    )
    return ap.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError(f"Config file is empty or invalid: {path}")
    return cfg


def _resolve_features_output_path(cfg: dict[str, Any]) -> Path:
    # Preferred modern run-dir layout used by extract_features.py
    if all(k in cfg for k in ("experiment_id", "results_root")):
        task_type = str(cfg.get("task_type", "feats"))
        return resolve_run_dir(cfg, task_type) / "features.npz"

    # Legacy mode (no run dir metadata)
    data_cfg = cfg.get("data") or {}
    output_path = data_cfg.get("output_path")
    if output_path:
        return Path(str(output_path)).expanduser()

    raise ValueError(
        "Could not infer features output path. Provide either "
        "(experiment_id + results_root [+ modality] [+ task_type]) or data.output_path."
    )


def _prepare_runtime_cfg(base_cfg: dict[str, Any], features_path: Path) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)

    data_cfg = cfg.setdefault("data", {})
    data_cfg["input_path"] = str(features_path)

    # Domain analysis defaults: if user didn't define explicit domains, auto-run pairwise
    # over all dataset labels present in the extracted NPZ.
    domain_cfg = cfg.get("domain_analysis")
    if isinstance(domain_cfg, dict):
        has_explicit_binary = "domain1" in domain_cfg and "domain2" in domain_cfg
        has_explicit_domains = "domains" in domain_cfg
        if not has_explicit_binary and not has_explicit_domains:
            domain_cfg.setdefault("mode", "pairwise_all")
            domain_cfg.setdefault("source_npz", str(features_path))
        elif domain_cfg.get("mode", "").strip().lower() in {"pairwise_all", "auto_pairwise"}:
            domain_cfg.setdefault("source_npz", str(features_path))
    return cfg


def _write_temp_cfg(cfg: dict[str, Any]) -> Path:
    tmp = tempfile.NamedTemporaryFile(prefix="feature_domain_pipeline_", suffix=".yaml", delete=False)
    with open(tmp.name, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return Path(tmp.name)


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"
    return env


def _run(cmd: list[str], *, dry_run: bool, env: dict[str, str]) -> None:
    print(f"[pipeline] $ {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.cfg).expanduser()
    base_cfg = _load_yaml(cfg_path)

    if args.skip_features and args.skip_pca and args.skip_domain:
        raise ValueError("All stages are skipped. Enable at least one stage.")

    features_path = _resolve_features_output_path(base_cfg)
    runtime_cfg = _prepare_runtime_cfg(base_cfg, features_path=features_path)
    runtime_cfg_path = _write_temp_cfg(runtime_cfg)
    env = _subprocess_env()

    print(f"[pipeline] base cfg   = {cfg_path}")
    print(f"[pipeline] runtime cfg = {runtime_cfg_path}")
    print(f"[pipeline] features    = {features_path}")

    try:
        py = sys.executable

        if not args.skip_features:
            cmd = [py, str(REPO_ROOT / "scripts" / "extract_features.py"), "--cfg", str(runtime_cfg_path)]
            if args.checkpoint:
                cmd.extend(["--checkpoint", str(Path(args.checkpoint).expanduser())])
            _run(cmd, dry_run=args.dry_run, env=env)
        else:
            print("[pipeline] skipping feature extraction")

        if not args.skip_pca:
            _run(
                [py, str(REPO_ROOT / "scripts" / "run_pca.py"), "--cfg", str(runtime_cfg_path)],
                dry_run=args.dry_run,
                env=env,
            )
        else:
            print("[pipeline] skipping PCA/UMAP")

        if not args.skip_domain:
            if "domain_analysis" not in runtime_cfg:
                print("[pipeline] domain_analysis section missing in config; skipping domain analysis")
            else:
                _run(
                    [py, str(REPO_ROOT / "scripts" / "run_domain_analysis.py"), "--cfg", str(runtime_cfg_path)],
                    dry_run=args.dry_run,
                    env=env,
                )
        else:
            print("[pipeline] skipping domain analysis")

        if not args.dry_run:
            print("[pipeline] completed")
    finally:
        if args.keep_temp_cfg:
            print(f"[pipeline] kept runtime config: {runtime_cfg_path}")
        else:
            try:
                runtime_cfg_path.unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[pipeline] ERROR: {exc}", file=sys.stderr)
        raise
