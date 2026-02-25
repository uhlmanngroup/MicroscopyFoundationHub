#!/usr/bin/env python3
"""Compute FID-style domain distance + LR separability from feature NPZ files.

Supports two modes:
1) Legacy binary mode (`domain1` + `domain2`) for backward compatibility.
2) Pairwise-all mode from a single NPZ (`source_npz`) that auto-detects datasets and
   computes all pairwise comparisons (works for 2, 3, or N datasets).

Example (legacy):
    python scripts/run_domain_analysis.py --cfg configs/mac/domain_analysis.yaml

Example (auto pairwise):
    python scripts/run_domain_analysis.py --cfg configs/mac/domain_analysis_triplet.yaml
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics as stats
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from dino_peft.analysis.dimred import load_feature_npz
from dino_peft.analysis.domain_metrics import (
    compute_fid_from_features,
    fit_pca_on_concat,
    load_features,
    lr_domain_separability,
)
from dino_peft.utils.paths import setup_run_dir, update_metrics, write_run_info

DEFAULT_CFG = Path(__file__).parent.parent / "configs" / "mac" / "domain_analysis.yaml"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Domain gap analysis based on DINO features.")
    ap.add_argument(
        "--cfg",
        "--config",
        dest="cfg",
        type=str,
        default=str(DEFAULT_CFG),
        help="Path to YAML config.",
    )
    return ap.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as handle:
        cfg = yaml.safe_load(handle)
    if not cfg:
        raise ValueError(f"Config file {path} is empty or invalid.")
    return cfg


def _normalize_sequence(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _resolve_output_paths(
    domain_cfg: Dict[str, Any],
    anchor_path: Path,
) -> tuple[Path, Path, Path]:
    """Return (output_dir, metrics_json_path, metrics_csv_path)."""
    legacy_out = domain_cfg.get("out_json")
    if legacy_out:
        json_path = Path(legacy_out)
        if not json_path.is_absolute():
            json_path = anchor_path.parent / json_path
        json_path = json_path.expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        csv_name = domain_cfg.get("csv_name")
        csv_path = (
            json_path.with_suffix(".csv")
            if not csv_name
            else (json_path.parent / csv_name)
        )
        csv_path = csv_path.expanduser()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        return json_path.parent, json_path, csv_path

    out_dir = domain_cfg.get("out_dir")
    if out_dir:
        out_dir = Path(out_dir)
        if not out_dir.is_absolute():
            out_dir = anchor_path.parent / out_dir
    else:
        subdir = domain_cfg.get("output_subdir", "domain-analysis")
        out_dir = anchor_path.parent / subdir
    out_dir = out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_name = domain_cfg.get("json_name", "metrics.json")
    csv_name = domain_cfg.get("csv_name", "metrics.csv")
    json_path = (out_dir / json_name).expanduser()
    csv_path = (out_dir / csv_name).expanduser()
    return out_dir, json_path, csv_path


def _fmt_filter(ids, names):
    parts = []
    if ids:
        parts.append(f"dataset_ids={ids}")
    if names:
        parts.append(f"dataset_names={names}")
    return ", ".join(parts) if parts else "all samples"


def _load_domain(entry: Dict[str, Any], *, default_label: str | None = None) -> Dict[str, Any]:
    if "npz" not in entry:
        raise ValueError("Each domain definition must include an 'npz' path.")
    path = Path(entry["npz"]).expanduser()
    dataset_ids = _normalize_sequence(entry.get("dataset_ids"))
    if dataset_ids is not None:
        dataset_ids = [int(v) for v in dataset_ids]
    dataset_names = _normalize_sequence(entry.get("dataset_names"))
    if dataset_names is not None:
        dataset_names = [str(v) for v in dataset_names]
    feats = load_features(path, dataset_ids=dataset_ids, dataset_names=dataset_names)

    label = entry.get("name")
    if not label:
        if dataset_names and len(dataset_names) == 1:
            label = dataset_names[0]
        elif dataset_ids and len(dataset_ids) == 1:
            label = f"id{dataset_ids[0]}"
        else:
            label = default_label or "domain"

    return {
        "label": str(label),
        "npz": path,
        "dataset_ids": dataset_ids,
        "dataset_names": dataset_names,
        "features": feats,
    }


def _discover_dataset_names(
    npz_path: Path,
    include_names: list[str] | None = None,
    exclude_names: list[str] | None = None,
) -> list[str]:
    bundle = load_feature_npz(npz_path)

    discovered: set[str] = set()
    if getattr(bundle, "dataset_names", None):
        discovered.update(str(v) for v in bundle.dataset_names if v is not None and str(v) != "")
    if bundle.meta and bundle.meta.get("dataset_name_to_id"):
        discovered.update(str(k) for k in bundle.meta["dataset_name_to_id"].keys())

    names = sorted(discovered)
    if not names:
        raise ValueError(
            f"Could not discover dataset names from {npz_path}. "
            "Ensure extract_features.py stored dataset_names metadata."
        )

    include = [str(v) for v in (include_names or [])]
    exclude = [str(v) for v in (exclude_names or [])]
    missing = sorted(set(include) - set(names))
    if missing:
        raise ValueError(f"Requested include_dataset_names not found in {npz_path}: {missing}")

    if include:
        names = [n for n in names if n in include]
    if exclude:
        names = [n for n in names if n not in set(exclude)]

    if len(names) < 2:
        raise ValueError(
            f"Need at least 2 datasets after filtering, got {names}. "
            "Adjust include/exclude filters."
        )
    return names


def _compute_pairwise_metrics(
    domains: list[Dict[str, Any]],
    *,
    n_components: int,
    seed: int,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for dom_a, dom_b in itertools.combinations(domains, 2):
        feats_a = dom_a["features"]
        feats_b = dom_b["features"]

        fdd_value = compute_fid_from_features(feats_a, feats_b)
        _, feats_a_pca, feats_b_pca = fit_pca_on_concat(
            feats_a,
            feats_b,
            n_components=n_components,
            random_state=seed,
        )
        lr_metrics = lr_domain_separability(
            feats_a_pca,
            feats_b_pca,
            random_state=seed,
        )

        row = {
            "domain_a": dom_a["label"],
            "domain_b": dom_b["label"],
            "domain_a_npz": str(dom_a["npz"]),
            "domain_b_npz": str(dom_b["npz"]),
            "domain_a_num_samples": int(feats_a.shape[0]),
            "domain_b_num_samples": int(feats_b.shape[0]),
            "domain_a_dataset_ids": json.dumps(dom_a.get("dataset_ids")),
            "domain_b_dataset_ids": json.dumps(dom_b.get("dataset_ids")),
            "domain_a_dataset_names": json.dumps(dom_a.get("dataset_names")),
            "domain_b_dataset_names": json.dumps(dom_b.get("dataset_names")),
            "frechet_dino_distance": float(fdd_value),
            "logreg_accuracy": float(lr_metrics["accuracy"]),
            "logreg_roc_auc": float(lr_metrics["roc_auc"]),
            "n_components": int(n_components),
            "seed": int(seed),
        }
        rows.append(row)

        print(
            "[domain_analysis] Pair "
            f"{dom_a['label']} vs {dom_b['label']}: "
            f"FDD={row['frechet_dino_distance']:.4f}, "
            f"LR_acc={row['logreg_accuracy'] * 100:.2f}%, "
            f"LR_auc={row['logreg_roc_auc'] * 100:.2f}"
        )
    return rows


def _pairwise_summary(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    def _mean_std(key: str) -> tuple[float | None, float | None]:
        vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
        if not vals:
            return None, None
        if len(vals) == 1:
            return vals[0], 0.0
        return float(stats.mean(vals)), float(stats.stdev(vals))

    fdd_mean, fdd_std = _mean_std("frechet_dino_distance")
    acc_mean, acc_std = _mean_std("logreg_accuracy")
    auc_mean, auc_std = _mean_std("logreg_roc_auc")
    return {
        "num_pairs": len(rows),
        "frechet_dino_distance_mean": fdd_mean,
        "frechet_dino_distance_std": fdd_std,
        "logreg_accuracy_mean": acc_mean,
        "logreg_accuracy_std": acc_std,
        "logreg_roc_auc_mean": auc_mean,
        "logreg_roc_auc_std": auc_std,
    }


def _write_kv_csv(path: Path, metrics: Dict[str, Any]) -> None:
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def _write_pairwise_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No pairwise rows to write.")
    fieldnames = [
        "domain_a",
        "domain_b",
        "domain_a_num_samples",
        "domain_b_num_samples",
        "frechet_dino_distance",
        "logreg_accuracy",
        "logreg_roc_auc",
        "n_components",
        "seed",
        "domain_a_npz",
        "domain_b_npz",
        "domain_a_dataset_ids",
        "domain_b_dataset_ids",
        "domain_a_dataset_names",
        "domain_b_dataset_names",
    ]
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mode_from_config(domain_cfg: Dict[str, Any]) -> str:
    # Explicit mode wins.
    explicit = str(domain_cfg.get("mode", "")).strip().lower()
    if explicit:
        if explicit in {"binary", "legacy_binary"}:
            return "binary"
        if explicit in {"pairwise_all", "auto_pairwise"}:
            return "pairwise_all"
        raise ValueError(f"Unsupported domain_analysis.mode='{explicit}'")

    has_binary = "domain1" in domain_cfg and "domain2" in domain_cfg
    has_source = "source_npz" in domain_cfg
    has_domains = "domains" in domain_cfg

    if has_binary:
        return "binary"
    if has_source or has_domains:
        return "pairwise_all"
    raise ValueError(
        "domain_analysis config must define either legacy 'domain1/domain2' "
        "or 'source_npz' (auto pairwise) or 'domains' (explicit list)."
    )


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.cfg).expanduser()
    print(f"[domain_analysis] Using config file: {cfg_path}")
    cfg = load_config(cfg_path)
    domain_cfg = cfg.get("domain_analysis") or {}

    if "n_components" not in domain_cfg or "seed" not in domain_cfg:
        raise ValueError("Config domain_analysis must define 'n_components' and 'seed'.")
    n_components = int(domain_cfg["n_components"])
    seed = int(domain_cfg["seed"])
    mode = _mode_from_config(domain_cfg)

    run_dir = None
    task_type = cfg.get("task_type", "domain-analysis")

    domains: list[Dict[str, Any]]
    anchor_path: Path

    if mode == "binary":
        required_keys = ["domain1", "domain2"]
        missing = [k for k in required_keys if k not in domain_cfg]
        if missing:
            raise ValueError(f"Config domain_analysis missing keys: {missing}")
        dom1 = _load_domain(domain_cfg["domain1"], default_label="domain1")
        dom2 = _load_domain(domain_cfg["domain2"], default_label="domain2")
        domains = [dom1, dom2]
        anchor_path = dom1["npz"]
    else:
        if "domains" in domain_cfg:
            entries = domain_cfg.get("domains") or []
            if not isinstance(entries, list) or len(entries) < 2:
                raise ValueError("domain_analysis.domains must be a list with at least 2 entries.")
            domains = [
                _load_domain(entry, default_label=f"domain{i+1}")
                for i, entry in enumerate(entries)
            ]
            anchor_path = domains[0]["npz"]
        else:
            source_npz = Path(str(domain_cfg["source_npz"])).expanduser()
            include_names = _normalize_sequence(domain_cfg.get("include_dataset_names"))
            exclude_names = _normalize_sequence(domain_cfg.get("exclude_dataset_names"))
            discovered = _discover_dataset_names(
                source_npz,
                include_names=[str(v) for v in include_names] if include_names else None,
                exclude_names=[str(v) for v in exclude_names] if exclude_names else None,
            )
            print(f"[domain_analysis] Auto-detected datasets in {source_npz}: {discovered}")
            domains = [
                _load_domain({"npz": str(source_npz), "dataset_names": [name], "name": name})
                for name in discovered
            ]
            anchor_path = source_npz

    if len(domains) < 2:
        raise ValueError(f"Need at least two domains, got {len(domains)}")

    output_dir, out_json, out_csv = _resolve_output_paths(domain_cfg, anchor_path)
    print(f"[domain_analysis] Output directory: {output_dir}")

    if "experiment_id" in cfg and "results_root" in cfg:
        run_dir = setup_run_dir(cfg, task_type=task_type, subdirs=None)
        run_info = {
            "task_type": task_type,
            "mode": mode,
            "pca_components": n_components,
            "seed": seed,
            "num_domains": len(domains),
            "domain_labels": [d["label"] for d in domains],
        }
        if mode == "binary":
            run_info["domain1_npz"] = str(domains[0]["npz"])
            run_info["domain2_npz"] = str(domains[1]["npz"])
        else:
            run_info["source_npz"] = str(anchor_path)
        write_run_info(run_dir, run_info)

    for idx, dom in enumerate(domains, start=1):
        feats = dom["features"]
        print(
            f"[domain_analysis] Domain{idx} '{dom['label']}': {dom['npz']} "
            f"(N={feats.shape[0]}, D={feats.shape[1]}; {_fmt_filter(dom.get('dataset_ids'), dom.get('dataset_names'))})"
        )

    # Legacy output shape preserved for exactly two domains in binary mode.
    if mode == "binary":
        pair_rows = _compute_pairwise_metrics(domains, n_components=n_components, seed=seed)
        row = pair_rows[0]
        payload = {
            "mode": "binary",
            "metrics": {
                "frechet_dino_distance": row["frechet_dino_distance"],
                "logreg_accuracy": row["logreg_accuracy"],
                "logreg_roc_auc": row["logreg_roc_auc"],
            },
            "config": {
                "domain1": {
                    "npz": str(domains[0]["npz"]),
                    "dataset_ids": domains[0]["dataset_ids"],
                    "dataset_names": domains[0]["dataset_names"],
                    "num_samples": int(domains[0]["features"].shape[0]),
                    "label": domains[0]["label"],
                },
                "domain2": {
                    "npz": str(domains[1]["npz"]),
                    "dataset_ids": domains[1]["dataset_ids"],
                    "dataset_names": domains[1]["dataset_names"],
                    "num_samples": int(domains[1]["features"].shape[0]),
                    "label": domains[1]["label"],
                },
                "n_components": n_components,
                "seed": seed,
            },
        }
        out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
        _write_kv_csv(
            out_csv,
            {
                "frechet_dino_distance": row["frechet_dino_distance"],
                "logreg_accuracy": row["logreg_accuracy"],
                "logreg_roc_auc": row["logreg_roc_auc"],
            },
        )
        print(f"[domain_analysis] Saved metrics JSON to {out_json}")
        print(f"[domain_analysis] Saved metrics CSV to {out_csv}")

        if run_dir is not None and domain_cfg.get("use_run_dir_metrics", False):
            update_metrics(
                run_dir,
                "domain_analysis",
                {
                    "frechet_dino_distance": row["frechet_dino_distance"],
                    "logreg_accuracy": row["logreg_accuracy"],
                    "logreg_roc_auc": row["logreg_roc_auc"],
                    "domain1_npz": str(domains[0]["npz"]),
                    "domain2_npz": str(domains[1]["npz"]),
                    "pca_components": n_components,
                },
            )
        return

    # Pairwise-all mode (N domains).
    pair_rows = _compute_pairwise_metrics(domains, n_components=n_components, seed=seed)
    summary = _pairwise_summary(pair_rows)

    payload = {
        "mode": "pairwise_all",
        "summary": summary,
        "config": {
            "n_components": n_components,
            "seed": seed,
            "domains": [
                {
                    "label": d["label"],
                    "npz": str(d["npz"]),
                    "dataset_ids": d["dataset_ids"],
                    "dataset_names": d["dataset_names"],
                    "num_samples": int(d["features"].shape[0]),
                }
                for d in domains
            ],
        },
        "pairwise_metrics": pair_rows,
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _write_pairwise_csv(out_csv, pair_rows)
    print(f"[domain_analysis] Saved pairwise metrics JSON to {out_json}")
    print(f"[domain_analysis] Saved pairwise metrics CSV to {out_csv}")

    # Also write a small aggregate CSV next to the pairwise table.
    summary_csv = out_csv.with_name(f"{out_csv.stem}_summary{out_csv.suffix}")
    _write_kv_csv(summary_csv, summary)
    print(f"[domain_analysis] Saved pairwise summary CSV to {summary_csv}")

    if run_dir is not None and domain_cfg.get("use_run_dir_metrics", False):
        update_metrics(
            run_dir,
            "domain_analysis",
            {
                "mode": "pairwise_all",
                "num_domains": len(domains),
                "num_pairs": summary["num_pairs"],
                "frechet_dino_distance_mean": summary["frechet_dino_distance_mean"],
                "logreg_accuracy_mean": summary["logreg_accuracy_mean"],
                "logreg_roc_auc_mean": summary["logreg_roc_auc_mean"],
                "anchor_npz": str(anchor_path),
                "pca_components": n_components,
            },
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[domain_analysis] ERROR: {exc}", file=sys.stderr)
        raise
