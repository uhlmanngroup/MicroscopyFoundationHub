#!/usr/bin/env python3
"""Five-seed foreground-IoU report for the October missing paper cells."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dino_peft.config import load_paths  # noqa: E402

CELLS = (
    "dinov2_triplet_head", "dinov2_triplet_lora", "dinov2_triplet_fullft",
    "openclip_aureus_lora", "openclip_coli_lora",
    "openclip_subtilis_lora", "openclip_triplet_lora",
    "dinov2_vnc_lora",
)
GROUPS = {"S. aureus": "jenilered", "E. coli": "pos", "B. subtilis": "test"}


def run_dir(scratch: Path, tag: str, cell: str, rep: int) -> Path:
    family = cell.split("_", 1)[0]
    modality = "em" if cell == "dinov2_vnc_lora" else "deepbacs"
    if cell == "dinov2_vnc_lora":
        task = "seg/single/droso-lora"
    elif "triplet" in cell:
        mode = cell.rsplit("_", 1)[-1]
        task = f"seg/triplet/coli-aureus-subtilis-{mode}"
    else:
        species = cell.split("_")[1]
        task = f"seg/single/{species}-lora"
    exp = f"{tag}_{cell}_seg_rep{rep}"
    return scratch / family / modality / task / exp


def expected_mode(cell: str) -> tuple[bool, bool]:
    mode = cell.rsplit("_", 1)[-1]
    return mode == "lora", mode == "fullft"


def expected_datasets(cell: str) -> dict[str, str]:
    if "triplet" in cell:
        return {"Pooled": "foreground_iou", **GROUPS}
    if cell == "dinov2_vnc_lora":
        return {"VNC": "foreground_iou"}
    species = cell.split("_")[1]
    label = {"aureus": "S. aureus", "coli": "E. coli", "subtilis": "B. subtilis"}[species]
    return {label: "foreground_iou"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--scratch-root", type=Path)
    args = ap.parse_args()
    scratch = args.scratch_root or Path(load_paths()["scratch_root"])
    output = scratch / "summaries" / args.run_tag
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    problems: list[str] = []
    for cell in CELLS:
        values: dict[str, dict[int, float]] = {name: {} for name in expected_datasets(cell)}
        for rep in range(1, 6):
            run = run_dir(scratch, args.run_tag, cell, rep)
            try:
                config = yaml.safe_load((run / "config_used.yaml").read_text())
                metrics = json.loads((run / "metrics.json").read_text())["eval"]
                if config.get("seed") != rep or config.get("split_seed") != rep + 100:
                    raise ValueError("seed or split_seed mismatch")
                if (bool(config.get("use_lora")), bool(config.get("full_finetune"))) != expected_mode(cell):
                    raise ValueError("tuning mode mismatch")
                if Path(config["results_root"]).resolve() != (scratch / cell.split("_", 1)[0]).resolve():
                    raise ValueError("results outside expected scratch family")
                if cell.startswith("dinov2_triplet"):
                    if config["backbone"]["variant"] != "large":
                        raise ValueError("DINOv2 triplet variant mismatch")
                    if config["lora_rank"] != 16 or config["lora_alpha"] != 32:
                        raise ValueError("DINOv2 triplet LoRA settings mismatch")
                if cell == "dinov2_vnc_lora":
                    if config["backbone"]["variant"] != "large":
                        raise ValueError("DINOv2 VNC variant mismatch")
                if cell.startswith("openclip"):
                    if config["backbone"]["variant"] != "ViT-L-14":
                        raise ValueError("OpenCLIP variant mismatch")
                group_metrics = metrics.get("foreground_iou_per_dataset") or {}
                rep_values = {}
                for name, key in expected_datasets(cell).items():
                    value = metrics.get(key) if key == "foreground_iou" else group_metrics.get(key)
                    if not isinstance(value, (int, float)):
                        raise ValueError(f"missing foreground IoU for {name}")
                    rep_values[name] = float(value)
                for name, value in rep_values.items():
                    values[name][rep] = value
            except (OSError, KeyError, TypeError, ValueError) as exc:
                problems.append(f"{cell} rep{rep}: {exc}")
        for name, seed_scores in values.items():
            scores = list(seed_scores.values())
            row = {"cell": cell, "dataset": name, "n": len(scores),
                   "mean": statistics.mean(scores) if scores else "",
                   "std": statistics.stdev(scores) if len(scores) > 1 else ""}
            row.update({f"seed{i}": seed_scores.get(i, "") for i in range(1, 6)})
            rows.append(row)
    with (output / "foreground_iou_mean_std.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["cell", "dataset", "n", "mean", "std", *(f"seed{i}" for i in range(1, 6))])
        writer.writeheader()
        writer.writerows(rows)
    status = {"complete": not problems, "expected_runs": len(CELLS) * 5,
              "problems": problems, "summary_csv": str(output / "foreground_iou_mean_std.csv")}
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    for row in rows:
        if row["n"] == 5:
            print(f"{row['cell']} | {row['dataset']} | {row['mean']:.3f} ± {row['std']:.3f}")
    if problems:
        print("Incomplete results:\n" + "\n".join(problems), file=sys.stderr)
        return 1
    print(f"Summary: {status['summary_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
