#!/usr/bin/env python3
"""Summarize exactly five corrected EM triplet seeds for each paper column."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dino_peft.config import load_config, load_paths  # noqa: E402


CELLS = {
    "dinov3_frozen": ("DINOv3 Frozen", "paired_dinov3_cluster.yaml", "seg/triplet-nolora"),
    "dinov3_fullft": ("DINOv3 FT", "paired_dinov3_cluster.yaml", "seg/triplet-fullft"),
    "openclip_frozen": ("OpenCLIP Frozen", "triplet_openclip_vitl14_cluster.yaml", "seg/triplet-head"),
    "openclip_fullft": ("OpenCLIP FT", "triplet_openclip_vitl14_cluster.yaml", "seg/triplet-fullft"),
    "resnet50_fullft": ("ResNet50 FT", "triplet_resnet50_fullft_cluster.yaml", "seg/triplet-fullft"),
}
GROUPS = {"VNC": "droso", "Kasthuri++": "kasthuri", "Lucchi++": "lucchi"}
ROWS = (*GROUPS, "Pooled foreground IoU", "Mean of three datasets")


def summarize(run_tag: str) -> tuple[dict[str, dict[str, list[float]]], Path]:
    repo = Path(__file__).resolve().parents[2]
    paths = load_paths()
    expected_data = str(Path(paths["data_root"]) / "EM" / "TRIPLET_LUCCHI_FIXED")
    values = {row: {cell: [] for cell in CELLS} for row in ROWS}

    for cell, (_, cfg_name, task_type) in CELLS.items():
        base_cfg = repo / "configs" / "cluster" / "EM" / cfg_name
        results_root = Path(load_config(base_cfg)["results_root"])
        for rep in range(1, 6):
            experiment_id = f"{run_tag}_em_triplet_{cell}_seg_rep{rep}"
            run_dir = results_root / "em" / task_type / experiment_id
            metrics_path = run_dir / "metrics.json"
            config_path = run_dir / "config_used.yaml"
            if not metrics_path.is_file() or not config_path.is_file():
                raise FileNotFoundError(f"Incomplete run: {run_dir}")
            config = yaml.safe_load(config_path.read_text())
            if config.get("seed") != rep or config.get("split_seed") != 100 + rep:
                raise ValueError(f"Unexpected seeds in {config_path}")
            if config.get("test_img_dir") != f"{expected_data}/test/images":
                raise ValueError(f"Run did not use corrected triplet: {config_path}")
            metrics = json.loads(metrics_path.read_text()).get("eval", {})
            per_dataset = metrics.get("foreground_iou_per_dataset") or {}
            if set(per_dataset) != set(GROUPS.values()):
                raise ValueError(f"Expected three dataset IoUs in {metrics_path}: {per_dataset}")
            for label, key in GROUPS.items():
                values[label][cell].append(float(per_dataset[key]))
            values["Pooled foreground IoU"][cell].append(float(metrics["foreground_iou"]))
            values["Mean of three datasets"][cell].append(
                statistics.mean(float(per_dataset[key]) for key in GROUPS.values())
            )

    output = Path(paths["scratch_root"]) / "triplet_lucchi_fixed_summary" / f"{run_tag}.csv"
    return values, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", required=True)
    args = parser.parse_args()
    values, output = summarize(args.run_tag)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Dataset"]
    for label, _, _ in CELLS.values():
        fields += [f"{label} mean", f"{label} std", f"{label} n"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in ROWS:
            record: dict[str, str | float | int] = {"Dataset": row}
            for cell, (label, _, _) in CELLS.items():
                samples = values[row][cell]
                record[f"{label} mean"] = statistics.mean(samples)
                record[f"{label} std"] = statistics.stdev(samples)
                record[f"{label} n"] = len(samples)
            writer.writerow(record)
    print(f"Saved five-seed summary: {output}")
    print("Pooled foreground IoU is computed across test pixels; it is not the mean of the three dataset IoUs.")


if __name__ == "__main__":
    main()
