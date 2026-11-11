#!/usr/bin/env python3
"""Evaluate frozen DINOv3-L single-dataset heads on other test datasets.

Each job loads one trained checkpoint and evaluates its head on the other two
datasets in the same modality. Inference is deterministic; the five trained
heads (reps 1-5) are the replicates used for mean and sample standard deviation.

Cluster examples:
    python scripts/eval_transferability.py --dry-run
    python scripts/eval_transferability.py --job-index 0   # one of 0..29
    python scripts/eval_transferability.py --aggregate-only

The default checkpoint layout matches the September 2026 frozen DINOv3-L
single-dataset sweeps. Override the roots for a copied set of runs/data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
EM_DATASETS = ("droso", "kasthuri", "lucchi")  # droso is the VNC stack
DEEPBACS_DATASETS = ("coli", "aureus", "subtilis")
DATASETS = {"em": EM_DATASETS, "deepbacs": DEEPBACS_DATASETS}
EM_FOLDER = {"droso": "Droso", "kasthuri": "Kasthuri++", "lucchi": "Lucchi++"}
EM_CONFIG = {
    "droso": "configs/cluster/EM/droso_dinov3_cluster.yaml",
    "kasthuri": "configs/cluster/EM/kasthuri_dinov3_head_cluster.yaml",
    "lucchi": "configs/cluster/EM/lucchi_dinov3_head_cluster.yaml",
}
DEEPBACS_CONFIG = "configs/cluster/deepbacs/deepbacs_single_dinov3_cluster.yaml"
METRICS = ("foreground_iou", "foreground_dice", "mean_iou", "mean_dice")
LONG_FIELDS = (
    "modality", "source", "target", "rep", "seed", "split_seed", "checkpoint",
    "test_img_dir", "test_mask_dir", "n_images", *METRICS,
)
SUMMARY_FIELDS = (
    "modality", "source", "target", "n", "n_images",
    *(f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "std")),
)


@dataclass(frozen=True)
class Job:
    modality: str
    source: str
    rep: int
    checkpoint: Path
    targets: tuple[str, ...]


def checkpoint_path(runs_root: Path, modality: str, source: str, rep: int) -> Path:
    """Use the exact frozen/head run names, excluding older seed=0 EM runs."""
    if modality == "em":
        return (
            runs_root / "em" / "seg" / f"{source}-nolora"
            / f"2026-09-10_em_{source}_dinov3_frozen_seg_rep{rep}"
            / "ckpts" / "best_model.pt"
        )
    return (
        runs_root / "deepbacs" / "seg" / "single" / f"{source}-head"
        / f"2026-09-08_deepbacs_{source}_dinov3_head_seg_rep{rep}"
        / "ckpts" / "best_model.pt"
    )


def build_jobs(
    modality: str,
    reps: Iterable[int],
    include_diagonal: bool,
    runs_root: Path,
) -> list[Job]:
    modalities = tuple(DATASETS) if modality == "both" else (modality,)
    if any(name not in DATASETS for name in modalities):
        raise ValueError(f"Unknown modality: {modality}")
    jobs = []
    for domain in modalities:
        for source in DATASETS[domain]:
            for rep in reps:
                targets = tuple(
                    target for target in DATASETS[domain]
                    if include_diagonal or target != source
                )
                jobs.append(Job(domain, source, int(rep),
                                checkpoint_path(Path(runs_root), domain, source, int(rep)), targets))
    return jobs


def target_paths(job: Job, target: str, em_data_root: Path, deepbacs_data_root: Path) -> tuple[Path, Path]:
    if job.modality == "em":
        root = em_data_root / EM_FOLDER[target]
        return root / "Test_In", root / "Test_Out"
    root = deepbacs_data_root / target / "test"
    return root / "images", root / "masks"


def result_path(out_dir: Path, job: Job, target: str) -> Path:
    return out_dir / "runs" / job.modality / job.source / f"rep{job.rep}" / f"{target}.json"


def _validate_result_row(row: Mapping, job: Job, target: str) -> None:
    for field, expected in (("modality", job.modality), ("source", job.source),
                            ("target", target), ("rep", job.rep)):
        if row.get(field) != expected:
            raise ValueError(f"Result {field}={row.get(field)!r}; expected {expected!r}")
    if "seed" not in row:
        raise ValueError(f"Result is missing training seed: {job.modality}/{job.source}/rep{job.rep}/{target}")
    int(row["seed"])
    if row.get("checkpoint") != str(job.checkpoint):
        raise ValueError(f"Result checkpoint does not match requested run: {job.checkpoint}")
    if int(row.get("n_images", 0)) <= 0:
        raise ValueError(f"Result has no images: {job.modality}/{job.source}/rep{job.rep}/{target}")
    for metric in METRICS:
        value = float(row[metric])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid {metric}={value} for {job.modality}/{job.source}/rep{job.rep}/{target}")


def summarize_results(rows: Iterable[Mapping], jobs: Iterable[Job]) -> list[dict]:
    """Require every expected directed pair/repetition, then compute sample SD."""
    jobs = list(jobs)
    expected = {(job.modality, job.source, target, job.rep): (job, target)
                for job in jobs for target in job.targets}
    by_key: dict[tuple, Mapping] = {}
    for row in rows:
        key = (row["modality"], row["source"], row["target"], int(row["rep"]))
        if key not in expected:
            raise ValueError(f"Unexpected transfer result: {key}")
        if key in by_key:
            raise ValueError(f"Duplicate transfer result: {key}")
        job, target = expected[key]
        _validate_result_row(row, job, target)
        by_key[key] = row
    missing = sorted(set(expected) - set(by_key))
    if missing:
        raise ValueError(f"Missing {len(missing)} transfer result(s), first: {missing[:5]}")

    groups: dict[tuple[str, str, str], list[Mapping]] = {}
    for key, row in by_key.items():
        groups.setdefault(key[:3], []).append(row)
    summary = []
    for (modality, source, target), group in sorted(groups.items()):
        seeds = {int(row["seed"]) for row in group}
        if len(seeds) != len(group):
            raise ValueError(f"Training seeds are not distinct: {modality}/{source}/{target}")
        image_counts = {int(row["n_images"]) for row in group}
        if len(image_counts) != 1:
            raise ValueError(f"Target image count changed across reps: {modality}/{source}/{target}: {image_counts}")
        result = {"modality": modality, "source": source, "target": target,
                  "n": len(group), "n_images": image_counts.pop()}
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(result)
    return summary


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def aggregate(out_dir: Path, jobs: list[Job]) -> None:
    rows = []
    for job in jobs:
        for target in job.targets:
            path = result_path(out_dir, job, target)
            if not path.is_file():
                raise FileNotFoundError(f"Missing transfer result: {path}")
            rows.append(json.loads(path.read_text()))
    summary = summarize_results(rows, jobs)
    _write_csv(out_dir / "transfer_runs.csv", LONG_FIELDS,
               sorted(rows, key=lambda row: (row["modality"], row["source"], row["target"], row["rep"])))
    _write_csv(out_dir / "transfer_summary.csv", SUMMARY_FIELDS, summary)
    for modality in sorted({job.modality for job in jobs}):
        by_pair = {(row["source"], row["target"]): row for row in summary if row["modality"] == modality}
        matrix = []
        for source in DATASETS[modality]:
            matrix.append({
                "source": source,
                **{
                    target: (
                        f"{by_pair[source, target]['foreground_iou_mean']:.4f} ± "
                        f"{by_pair[source, target]['foreground_iou_std']:.4f}"
                        if (source, target) in by_pair else ""
                    )
                    for target in DATASETS[modality]
                },
            })
        _write_csv(out_dir / f"transfer_matrix_{modality}_foreground_iou.csv",
                   ("source", *DATASETS[modality]), matrix)
    print(f"Wrote transfer runs, summary, and foreground-IoU matrices in {out_dir}")


def _target_config(job: Job, target: str, source_cfg: Mapping,
                   em_data_root: Path, deepbacs_data_root: Path) -> dict:
    """Target pairing/mask protocol, with the source model's image preprocessing."""
    from dino_peft.config import load_config

    config_file = EM_CONFIG[target] if job.modality == "em" else DEEPBACS_CONFIG
    base_cfg = load_config(REPO_ROOT / config_file)
    image_dir, mask_dir = target_paths(job, target, em_data_root, deepbacs_data_root)
    return compose_target_config(base_cfg, source_cfg, image_dir, mask_dir, job.modality)


def compose_target_config(base_cfg: Mapping, source_cfg: Mapping,
                          image_dir: Path, mask_dir: Path, modality: str) -> dict:
    """Keep target data rules while matching the source model's image pipeline."""
    cfg = dict(base_cfg)
    cfg["test_img_dir"], cfg["test_mask_dir"] = str(image_dir), str(mask_dir)
    cfg["img_size"] = source_cfg["img_size"]
    cfg["clahe_norm"] = bool(source_cfg.get("clahe_norm", False))
    if str(cfg.get("modality", "em")) != modality:
        raise ValueError(f"Target config modality mismatch: expected {modality}")
    if int(cfg["num_classes"]) != int(source_cfg["num_classes"]):
        raise ValueError("Source/target num_classes mismatch")
    return cfg


def _check_complete_pairs(dataset, image_dir: Path, mask_dir: Path, recursive: bool) -> None:
    from dino_peft.datasets.paired_dirs_seg import _list_files

    images = _list_files(image_dir, recursive)
    masks = _list_files(mask_dir, recursive)
    paired_images = [image for image, _ in dataset.pairs]
    paired_masks = [mask for _, mask in dataset.pairs]
    if (len(paired_images) != len(images) or len(paired_masks) != len(masks)
            or len(set(paired_images)) != len(images) or len(set(paired_masks)) != len(masks)):
        raise ValueError(
            f"Incomplete image/mask pairing: images={len(images)}, masks={len(masks)}, "
            f"pairs={len(dataset.pairs)} in {image_dir} / {mask_dir}"
        )


def _load_frozen_model(job: Job, device, weights_override: str | None,
                       repo_override: str | None):
    import torch
    from dino_peft.backbones import build_backbone, resolve_backbone_cfg
    from dino_peft.models.head_seg1x1 import SegHeadDeconv

    ckpt = torch.load(job.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg")
    if not isinstance(cfg, dict):
        raise ValueError(f"Checkpoint lacks training cfg: {job.checkpoint}")
    bb_cfg = resolve_backbone_cfg(cfg)
    if (str(bb_cfg["name"]).lower(), str(bb_cfg["variant"]).lower()) != ("dinov3", "vitl16"):
        raise ValueError(f"Expected DINOv3-L/16, found {bb_cfg['name']}/{bb_cfg['variant']} in {job.checkpoint}")
    if (cfg.get("use_lora") or cfg.get("full_finetune")
            or (cfg.get("lora") or {}).get("enabled") or ckpt.get("backbone_lora")
            or ckpt.get("backbone")):
        raise ValueError(f"Expected a frozen backbone checkpoint: {job.checkpoint}")
    if "seed" not in cfg:
        raise ValueError(f"Checkpoint lacks training seed: {job.checkpoint}")
    if cfg.get("experiment_id") != job.checkpoint.parents[1].name:
        raise ValueError(f"Checkpoint experiment_id does not match its run directory: {job.checkpoint}")
    if str(cfg.get("modality", "em")).lower() != job.modality:
        raise ValueError(f"Checkpoint modality does not match {job.modality}: {job.checkpoint}")
    if int(cfg.get("num_classes", -1)) != 2:
        raise ValueError(f"Expected binary segmentation checkpoint: {job.checkpoint}")
    if "head" not in ckpt:
        raise ValueError(f"Checkpoint lacks head weights: {job.checkpoint}")
    if weights_override:
        bb_cfg["weights"] = weights_override
    if repo_override:
        bb_cfg["repo_dir"] = repo_override
    backbone = build_backbone(bb_cfg, device=device)
    head = SegHeadDeconv(in_ch=backbone.embed_dim, num_classes=2, n_ups=4, base_ch=512).to(device)
    head.load_state_dict(ckpt["head"], strict=True)
    backbone.eval()
    head.eval()
    return backbone, head, cfg


def evaluate_job(job: Job, out_dir: Path, em_data_root: Path, deepbacs_data_root: Path,
                 num_workers: int, device_name: str, weights_override: str | None,
                 repo_override: str | None) -> None:
    import torch
    from torch.utils.data import DataLoader
    from dino_peft.utils.transforms import em_seg_transforms
    from scripts.eval_em_seg import build_dataset_from_cfg, eval_loop, pad_collate

    if not job.checkpoint.is_file():
        raise FileNotFoundError(f"Missing frozen checkpoint: {job.checkpoint}")
    if device_name == "auto":
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if mps_available else "cpu")
    else:
        device = torch.device(device_name)
    backbone, head, source_cfg = _load_frozen_model(job, device, weights_override, repo_override)
    print(f"[{job.modality}/{job.source}/rep{job.rep}] checkpoint={job.checkpoint} device={device}")

    for target in job.targets:
        target_cfg = _target_config(job, target, source_cfg, em_data_root, deepbacs_data_root)
        image_dir, mask_dir = target_paths(job, target, em_data_root, deepbacs_data_root)
        transform = em_seg_transforms(clahe_norm=target_cfg["clahe_norm"])
        dataset = build_dataset_from_cfg(target_cfg, "test", transform)
        recursive = bool((target_cfg.get("dataset", {}).get("params") or {}).get("recursive", False))
        _check_complete_pairs(dataset, image_dir, mask_dir, recursive)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                            pin_memory=(device.type == "cuda"), collate_fn=pad_collate)
        with torch.inference_mode():
            iou, dice, fg_iou, fg_dice, _ = eval_loop(
                backbone, head, loader, device, 2, out_dir=None, expected_groups=None,
            )
        row = {
            "modality": job.modality, "source": job.source, "target": target,
            "rep": job.rep, "seed": int(source_cfg["seed"]),
            "split_seed": int(source_cfg["split_seed"]) if "split_seed" in source_cfg else None,
            "checkpoint": str(job.checkpoint), "n_images": len(dataset),
            "test_img_dir": str(image_dir), "test_mask_dir": str(mask_dir),
            "foreground_iou": float(fg_iou), "foreground_dice": float(fg_dice),
            "mean_iou": float(iou.mean()), "mean_dice": float(dice.mean()),
        }
        _validate_result_row(row, job, target)
        path = result_path(out_dir, job, target)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
        os.replace(temp, path)
        print(f"[{job.modality}/{job.source} -> {target}/rep{job.rep}] "
              f"n={len(dataset)} IoU_fg={fg_iou:.4f} Dice_fg={fg_dice:.4f} -> {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modality", choices=("em", "deepbacs", "both"), default="both")
    parser.add_argument("--reps", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--runs-root", type=Path, help="Root containing em/ and deepbacs/ frozen runs; default ${scratch_root}/dinov3")
    parser.add_argument("--data-root", type=Path, help="Default data root from configs/paths.yaml")
    parser.add_argument("--em-data-root", type=Path, help="EM root containing Droso/, Kasthuri++/, Lucchi++/")
    parser.add_argument("--deepbacs-data-root", type=Path, help="DeepBacs single/ root containing coli/, aureus/, subtilis/")
    parser.add_argument("--out-dir", type=Path, help="Separate transfer results directory; default <runs-root>/transferability/dinov3_frozen")
    parser.add_argument("--job-index", type=int, help="Run one source/rep job (0..29 for both modalities, five reps)")
    parser.add_argument("--include-diagonal", action="store_true", help="Also score each model on its own test set")
    parser.add_argument("--aggregate-only", action="store_true", help="Summarize completed inference results without loading models")
    parser.add_argument("--dry-run", action="store_true", help="Print jobs, paths, and missing checkpoints/data without inference")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--backbone-weights", help="Override pretrained DINOv3-L weights path embedded in checkpoints")
    parser.add_argument("--dinov3-repo-dir", help="Override DINOv3 repository path embedded in checkpoints")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(set(args.reps)) != len(args.reps) or any(rep < 1 for rep in args.reps):
        raise ValueError("--reps must contain distinct positive repetition numbers")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    if args.job_index is not None and args.aggregate_only:
        raise ValueError("--job-index and --aggregate-only cannot be combined")
    requested_modalities = tuple(DATASETS) if args.modality == "both" else (args.modality,)
    needs_data_paths = (
        not args.aggregate_only and args.data_root is None
        and (("em" in requested_modalities and args.em_data_root is None)
             or ("deepbacs" in requested_modalities and args.deepbacs_data_root is None))
    )
    if args.runs_root is None or needs_data_paths:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from dino_peft.config import load_paths
        paths = load_paths()
    else:
        paths = {}
    runs_root = args.runs_root or Path(paths["scratch_root"]) / "dinov3"
    data_root = args.data_root or (Path(paths["data_root"]) if paths else None)
    em_data_root = args.em_data_root or (data_root / "EM" if data_root else Path("/unused/EM"))
    deepbacs_data_root = args.deepbacs_data_root or (data_root / "deepbacs" / "single" if data_root else Path("/unused/deepbacs"))
    out_dir = args.out_dir or runs_root / "transferability" / "dinov3_frozen"
    jobs = build_jobs(args.modality, args.reps, args.include_diagonal, runs_root)

    if args.aggregate_only:
        aggregate(out_dir, jobs)
        return 0
    if args.job_index is not None:
        if not 0 <= args.job_index < len(jobs):
            raise ValueError(f"--job-index must be in 0..{len(jobs)-1}")
        jobs = [jobs[args.job_index]]
    if args.dry_run:
        for index, job in enumerate(build_jobs(args.modality, args.reps, args.include_diagonal, runs_root)):
            if args.job_index is not None and index != args.job_index:
                continue
            status = "OK" if job.checkpoint.is_file() else "MISSING CHECKPOINT"
            targets = ", ".join(
                f"{target} ({'OK' if all(p.is_dir() for p in target_paths(job, target, em_data_root, deepbacs_data_root)) else 'MISSING DATA'})"
                for target in job.targets
            )
            print(f"{index:02d} {job.modality}/{job.source}/rep{job.rep}: {status}; targets: {targets}\n    {job.checkpoint}")
        print(f"Output: {out_dir}")
        return 0
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT))
    for job in jobs:
        evaluate_job(job, out_dir, em_data_root, deepbacs_data_root,
                     args.num_workers, args.device, args.backbone_weights, args.dinov3_repo_dir)
    if args.job_index is None:
        aggregate(out_dir, jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
