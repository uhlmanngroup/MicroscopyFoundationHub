#!/usr/bin/env python3
"""Summarize segmentation experiment runs into CSV/JSON reports.

Example (local):
    python scripts/analysis/summarize_seg_results.py --root /path/to/results/seg

Example (cluster):
    sbatch slurm/summarize_seg_results.sbatch
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as stats
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from dino_peft.backbones import resolve_backbone_cfg
from dino_peft.models.lora import resolve_full_finetune
METRIC_KEYS: Tuple[str, ...] = (
    "mean_dice",
    "mean_iou",
    "foreground_dice",
    "foreground_iou",
)


@dataclass
class RunRecord:
    """Container for per-run metrics and metadata."""

    run_dir: Path
    rel_run_dir: str
    experiment_id: str
    modality: Optional[str]
    task_type: Optional[str]
    backbone_name: Optional[str]
    backbone_variant: Optional[str]
    dino_size: str
    use_lora: bool
    full_finetune: bool
    training_mode: str
    replicate: Optional[int]
    seed: Optional[int]
    loss: Optional[str]
    dataset_label: Optional[str]
    dataset_type: Optional[str]
    pair_mode: Optional[str]
    mean_dice: Optional[float]
    mean_iou: Optional[float]
    foreground_dice: Optional[float]
    foreground_iou: Optional[float]
    best_val_loss: Optional[float]
    best_epoch: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate segmentation experiment metrics across DINO sizes "
            "and LoRA configurations."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Experiment directory that contains run folders.",
    )
    parser.add_argument(
        "--summary-dir-name",
        default="summary",
        help="Name of the folder that will store the generated reports.",
    )
    parser.add_argument(
        "--config-candidates",
        nargs="+",
        default=("config_used.yaml", "config.yaml"),
        help="Config filenames to probe in each run directory.",
    )
    parser.add_argument(
        "--metrics-filename",
        default="metrics.json",
        help="Name of the metrics file inside each run directory.",
    )
    return parser.parse_args()


def safe_relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def find_config(run_dir: Path, candidates: Sequence[str]) -> Optional[Path]:
    for candidate in candidates:
        cfg_path = run_dir / candidate
        if cfg_path.is_file():
            return cfg_path
    return None


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def read_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def extract_replicate(name: str) -> Optional[int]:
    match = re.search(r"rep(\d+)", name)
    if match:
        return int(match.group(1))
    return None


def size_rank(size: Optional[str]) -> int:
    order = {"tiny": 0, "small": 1, "base": 2, "large": 3, "giant": 4, "huge": 5}
    return order.get((size or "").lower(), len(order))


def infer_training_mode(use_lora: bool, full_finetune: bool) -> str:
    if use_lora and full_finetune:
        return "invalid_lora_and_fullft"
    if use_lora:
        return "lora"
    if full_finetune:
        return "full_finetune"
    return "head_only"


def _canonical_dataset_label(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    s = str(text).strip().lower()
    if not s:
        return None
    if "kasthuri" in s:
        return "Kasthuri++"
    if "lucchi" in s:
        return "Lucchi++"
    if "droso" in s or "drosophila" in s or "vnc" in s:
        return "VNC"
    if "triplet" in s:
        return "Triplet"
    if "paired" in s:
        return "Paired"
    if "deepbacs" in s:
        return "DeepBacs"
    return None


def infer_dataset_label(cfg: Dict, run_dir: Path) -> str:
    dataset_cfg = cfg.get("dataset") or {}
    dataset_params = dataset_cfg.get("params") or {}

    explicit_label = dataset_cfg.get("label") or dataset_params.get("label")
    explicit_canonical = _canonical_dataset_label(explicit_label)
    if explicit_canonical:
        return explicit_canonical
    if explicit_label and str(explicit_label).strip():
        return str(explicit_label).strip()

    candidates = [
        cfg.get("task_type"),
        cfg.get("experiment_id"),
        cfg.get("train_img_dir"),
        cfg.get("train_mask_dir"),
        cfg.get("test_img_dir"),
        cfg.get("test_mask_dir"),
        str(run_dir),
    ]
    for item in candidates:
        label = _canonical_dataset_label(item)
        if label:
            return label

    fallback = _canonical_dataset_label(dataset_cfg.get("type"))
    if fallback:
        return fallback
    ds_type = str(dataset_cfg.get("type") or "Unknown").strip()
    return ds_type.title() if ds_type else "Unknown"


def collect_runs(
    root: Path, summary_dir: Path, metrics_filename: str, config_candidates: Sequence[str]
) -> List[RunRecord]:
    run_records: List[RunRecord] = []
    for metrics_path in sorted(root.rglob(metrics_filename)):
        if summary_dir in metrics_path.parents:
            continue
        run_dir = metrics_path.parent
        cfg_path = find_config(run_dir, config_candidates)
        if cfg_path is None:
            print(f"[warn] config missing for {run_dir}", flush=True)
            continue
        try:
            metrics = read_json(metrics_path)
        except json.JSONDecodeError as err:
            print(f"[warn] skipping {metrics_path}: {err}", flush=True)
            continue
        try:
            cfg = read_yaml(cfg_path) or {}
        except yaml.YAMLError as err:
            print(f"[warn] skipping {cfg_path}: {err}", flush=True)
            continue

        eval_metrics = metrics.get("eval", {})
        train_metrics = metrics.get("train", {})

        backbone_cfg = resolve_backbone_cfg(cfg)
        run_records.append(
            RunRecord(
                run_dir=run_dir,
                rel_run_dir=safe_relative_to(run_dir, root),
                experiment_id=cfg.get("experiment_id", run_dir.name),
                modality=str(cfg.get("modality", "em")).strip().lower() or "em",
                task_type=cfg.get("task_type"),
                backbone_name=backbone_cfg.get("name"),
                backbone_variant=backbone_cfg.get("variant"),
                dino_size=str(backbone_cfg.get("variant", cfg.get("dino_size", "unknown"))).lower(),
                use_lora=bool(cfg.get("use_lora", False)),
                full_finetune=bool(resolve_full_finetune(cfg)),
                training_mode=infer_training_mode(
                    use_lora=bool(cfg.get("use_lora", False)),
                    full_finetune=bool(resolve_full_finetune(cfg)),
                ),
                replicate=extract_replicate(run_dir.name),
                seed=train_metrics.get("seed"),
                loss=cfg.get("loss"),
                dataset_label=infer_dataset_label(cfg, run_dir),
                dataset_type=(cfg.get("dataset") or {}).get("type"),
                pair_mode=((cfg.get("dataset") or {}).get("params") or {}).get(
                    "pair_mode"
                ),
                mean_dice=eval_metrics.get("mean_dice"),
                mean_iou=eval_metrics.get("mean_iou"),
                foreground_dice=eval_metrics.get("foreground_dice"),
                foreground_iou=eval_metrics.get("foreground_iou"),
                best_val_loss=train_metrics.get("best_val_loss"),
                best_epoch=train_metrics.get("best_epoch"),
            )
        )
    return run_records


def compute_stats(values: Iterable[float]) -> Tuple[Optional[float], Optional[float]]:
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return None, None
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(stats.mean(vals)), float(stats.stdev(vals))


def build_aggregates(run_records: List[RunRecord]) -> List[Dict]:
    grouped: Dict[Tuple[str, str, str, str, bool, bool, str], Dict[str, List[float]]] = {}
    counts: Dict[Tuple[str, str, str, str, bool, bool, str], int] = {}

    for record in run_records:
        ds_label = str(record.dataset_label or "Unknown")
        modality = str(record.modality or "em").lower()
        task_type = str(record.task_type or "unknown")
        ds_type = (record.dataset_type or "unknown").lower()
        key = (
            ds_label,
            modality,
            task_type,
            ds_type,
            record.use_lora,
            record.full_finetune,
            record.dino_size,
        )
        if key not in grouped:
            grouped[key] = {metric: [] for metric in METRIC_KEYS}
            counts[key] = 0
        counts[key] += 1
        for metric in METRIC_KEYS:
            value = getattr(record, metric)
            if isinstance(value, (int, float)):
                grouped[key][metric].append(float(value))

    mode_order = {
        "head_only": 0,
        "lora": 1,
        "full_finetune": 2,
        "invalid_lora_and_fullft": 3,
    }

    def _sort_key(
        item: Tuple[Tuple[str, str, str, str, bool, bool, str], Dict[str, List[float]]]
    ):
        (ds_label, modality, task_type, ds_type, use_lora, full_finetune, size), _ = item
        mode = infer_training_mode(use_lora, full_finetune)
        return (
            modality,
            ds_label.lower(),
            task_type,
            ds_type,
            mode_order.get(mode, 99),
            size_rank(size),
            size,
        )

    aggregates: List[Dict] = []
    for key, metric_lists in sorted(grouped.items(), key=_sort_key):
        dataset_label, modality, task_type, dataset_type, use_lora, full_finetune, size = key
        row: Dict[str, Optional[float]] = {
            "dataset_label": dataset_label,
            "modality": modality,
            "task_type": task_type,
            "dataset_type": dataset_type,
            "use_lora": use_lora,
            "full_finetune": full_finetune,
            "training_mode": infer_training_mode(use_lora, full_finetune),
            "dino_size": size,
            "num_runs": counts.get(key, 0),
        }
        for metric, values in metric_lists.items():
            mean_val, std_val = compute_stats(values)
            row[f"{metric}_mean"] = mean_val
            row[f"{metric}_std"] = std_val
        aggregates.append(row)
    return aggregates


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def serialize_run_record(record: RunRecord) -> Dict[str, object]:
    return {
        "run_dir": str(record.run_dir),
        "rel_run_dir": record.rel_run_dir,
        "experiment_id": record.experiment_id,
        "modality": record.modality,
        "task_type": record.task_type,
        "backbone_name": record.backbone_name,
        "backbone_variant": record.backbone_variant,
        "dino_size": record.dino_size,
        "use_lora": record.use_lora,
        "full_finetune": record.full_finetune,
        "training_mode": record.training_mode,
        "replicate": record.replicate,
        "seed": record.seed,
        "loss": record.loss,
        "dataset_label": record.dataset_label,
        "dataset_type": record.dataset_type,
        "pair_mode": record.pair_mode,
        "mean_dice": record.mean_dice,
        "mean_iou": record.mean_iou,
        "foreground_dice": record.foreground_dice,
        "foreground_iou": record.foreground_iou,
        "best_val_loss": record.best_val_loss,
        "best_epoch": record.best_epoch,
    }


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().absolute()
    if not root.exists():
        raise SystemExit(f"root directory not found: {root}")
    summary_dir = root / args.summary_dir_name
    summary_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] scanning runs under {root}")
    run_records = collect_runs(
        root=root,
        summary_dir=summary_dir,
        metrics_filename=args.metrics_filename,
        config_candidates=args.config_candidates,
    )
    if not run_records:
        print("[warn] no runs discovered, nothing to summarize.")
        return
    aggregates = build_aggregates(run_records)

    run_fieldnames = [
        "run_dir",
        "rel_run_dir",
        "experiment_id",
        "modality",
        "task_type",
        "use_lora",
        "full_finetune",
        "training_mode",
        "backbone_name",
        "backbone_variant",
        "dino_size",
        "replicate",
        "seed",
        "loss",
        "dataset_label",
        "dataset_type",
        "pair_mode",
        "mean_dice",
        "mean_iou",
        "foreground_dice",
        "foreground_iou",
        "best_val_loss",
        "best_epoch",
    ]
    summary_fieldnames = [
        "dataset_label",
        "modality",
        "dataset_type",
        "task_type",
        "use_lora",
        "full_finetune",
        "training_mode",
        "dino_size",
        "num_runs",
        "mean_dice_mean",
        "mean_dice_std",
        "mean_iou_mean",
        "mean_iou_std",
        "foreground_dice_mean",
        "foreground_dice_std",
        "foreground_iou_mean",
        "foreground_iou_std",
    ]

    run_rows = [serialize_run_record(record) for record in run_records]
    write_csv(summary_dir / "run_metrics.csv", run_rows, run_fieldnames)
    write_csv(summary_dir / "summary.csv", aggregates, summary_fieldnames)

    manifest = {
        "root": str(root),
        "summary_dir": str(summary_dir),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "num_runs": len(run_records),
        "run_metrics_csv": str(summary_dir / "run_metrics.csv"),
        "summary_csv": str(summary_dir / "summary.csv"),
        "aggregate_metrics": aggregates,
    }
    with (summary_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"[info] wrote {summary_dir / 'run_metrics.csv'}")
    print(f"[info] wrote {summary_dir / 'summary.csv'}")
    print(f"[info] wrote {summary_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
