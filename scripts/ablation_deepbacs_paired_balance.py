#!/usr/bin/env python3
"""Run balanced-vs-unbalanced paired DeepBacs training.

Example (local):
    python scripts/ablation_deepbacs_paired_balance.py \
      --cfg configs/cluster/deepbacs/deepbacs_paired_balanced_dinov2_cluster.yaml \
      --variant both
"""
import argparse
import random
from copy import deepcopy
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from dino_peft.trainers.seg_trainer import SegTrainer

SOURCE_ALIASES = {
    "aureus": ("aureus", "jenilered"),
    "coli": ("coli", "pos"),
    "subtilis": ("subtilis", "test"),
}

PAIR_SOURCE_PATTERNS = {
    "aureus-subtilis": ("aureus", "subtilis"),
    "coli-aureus": ("coli", "aureus"),
    "coli-subtilis": ("coli", "subtilis"),
}


class BalancedPairBatchSampler:
    """Yield one sample from each configured group per batch."""

    def __init__(self, group_a_indices, group_b_indices, seed=0):
        self.group_a_indices = list(group_a_indices)
        self.group_b_indices = list(group_b_indices)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        group_a = self.group_a_indices[:]
        group_b = self.group_b_indices[:]
        rng.shuffle(group_a)
        rng.shuffle(group_b)
        n = min(len(group_a), len(group_b))
        for i in range(n):
            yield [group_a[i], group_b[i]]
        self.epoch += 1

    def __len__(self):
        return min(len(self.group_a_indices), len(self.group_b_indices))


def _load_cfg(path: Path):
    with path.open("r") as f:
        return yaml.safe_load(f)


def _normalize_patterns(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value if str(v).strip()]


def _parse_balanced_sampling_cfg(cfg: dict):
    sampling_cfg = dict(cfg.get("balanced_sampling") or {})
    match_source = str(sampling_cfg.get("match_source", "path")).strip().lower() or "path"
    if match_source not in {"path", "stem", "name"}:
        raise ValueError(f"Unsupported balanced_sampling.match_source '{match_source}'")

    group_a = dict(sampling_cfg.get("group_a") or {})
    group_b = dict(sampling_cfg.get("group_b") or {})
    group_a_patterns = [p.lower() for p in _normalize_patterns(group_a.get("patterns"))]
    group_b_patterns = [p.lower() for p in _normalize_patterns(group_b.get("patterns"))]

    if group_a_patterns and group_b_patterns:
        group_a_name = str(group_a.get("name", "group_a")).strip() or "group_a"
        group_b_name = str(group_b.get("name", "group_b")).strip() or "group_b"
        return {
            "group_a_name": group_a_name,
            "group_b_name": group_b_name,
            "group_a_patterns": group_a_patterns,
            "group_b_patterns": group_b_patterns,
            "match_source": match_source,
            "pair_key": None,
        }

    pair_key = _infer_pair_key(cfg)
    group_a_name, group_b_name = _resolve_pair_sources(pair_key)
    return {
        "group_a_name": group_a_name,
        "group_b_name": group_b_name,
        "group_a_patterns": [p.lower() for p in SOURCE_ALIASES.get(group_a_name, (group_a_name,))],
        "group_b_patterns": [p.lower() for p in SOURCE_ALIASES.get(group_b_name, (group_b_name,))],
        "match_source": match_source,
        "pair_key": pair_key,
    }


def _canonicalize_pair_key(value: str) -> str:
    key = str(value).strip().lower().replace("_", "-")
    if key in PAIR_SOURCE_PATTERNS:
        return key
    parts = [part for part in key.split("-") if part]
    if len(parts) == 2:
        reversed_key = "-".join(reversed(parts))
        if reversed_key in PAIR_SOURCE_PATTERNS:
            return reversed_key
    raise ValueError(
        f"Unknown DeepBacs paired dataset '{value}'. "
        f"Expected one of: {', '.join(sorted(PAIR_SOURCE_PATTERNS.keys()))}"
    )


def _infer_pair_key(cfg: dict) -> str:
    explicit = cfg.get("paired_dataset_name") or cfg.get("paired_dataset")
    if explicit:
        return _canonicalize_pair_key(explicit)

    candidate_paths = [
        cfg.get("train_img_dir"),
        cfg.get("train_mask_dir"),
        cfg.get("test_img_dir"),
        cfg.get("test_mask_dir"),
        cfg.get("task_type"),
        cfg.get("experiment_id"),
    ]
    known_keys = list(PAIR_SOURCE_PATTERNS.keys())
    for candidate in candidate_paths:
        if not candidate:
            continue
        text = str(candidate).lower().replace("_", "-")
        for key in known_keys:
            if key in text:
                return key
            parts = key.split("-")
            if len(parts) == 2 and f"{parts[1]}-{parts[0]}" in text:
                return key

    raise ValueError(
        "Could not infer the DeepBacs paired dataset from config. "
        "Set paired_dataset_name to one of: "
        + ", ".join(sorted(PAIR_SOURCE_PATTERNS.keys()))
    )


def _resolve_pair_sources(pair_key: str):
    canonical = _canonicalize_pair_key(pair_key)
    return PAIR_SOURCE_PATTERNS[canonical]


def _match_text(img_path: Path, match_source: str) -> str:
    if match_source in {"stem", "name"}:
        return img_path.stem.lower()
    return img_path.as_posix().lower()


def _partition_indices_by_rules(pairs, sampling_cfg: dict):
    group_a_name = sampling_cfg["group_a_name"]
    group_b_name = sampling_cfg["group_b_name"]
    group_a_patterns = sampling_cfg["group_a_patterns"]
    group_b_patterns = sampling_cfg["group_b_patterns"]
    match_source = sampling_cfg["match_source"]

    group_a_idx = []
    group_b_idx = []
    unknown = []
    ambiguous = []

    for i, (img_path, _) in enumerate(pairs):
        text = _match_text(img_path, match_source)
        in_group_a = any(pattern in text for pattern in group_a_patterns)
        in_group_b = any(pattern in text for pattern in group_b_patterns)
        if in_group_a and in_group_b:
            ambiguous.append(img_path.name)
        elif in_group_a:
            group_a_idx.append(i)
        elif in_group_b:
            group_b_idx.append(i)
        else:
            unknown.append(img_path.name)

    if ambiguous:
        preview = ", ".join(ambiguous[:5])
        raise ValueError(
            "Some training samples matched both balanced groups. "
            f"Examples: {preview}"
        )
    if unknown:
        preview = ", ".join(unknown[:5])
        raise ValueError(
            "Could not assign some training samples to either balanced group. "
            f"Examples: {preview}"
        )
    if not group_a_idx or not group_b_idx:
        raise ValueError(
            "Balanced sampling requires samples from both groups; "
            f"got {group_a_name}={len(group_a_idx)} {group_b_name}={len(group_b_idx)}"
        )

    return group_a_idx, group_b_idx


def _apply_tuning_mode(cfg: dict, tuning_mode: str):
    mode = str(tuning_mode).strip().lower()
    if mode == "cfg":
        return "lora" if cfg.get("use_lora") else "fullft" if cfg.get("full_finetune") else "frozen"
    if mode == "frozen":
        cfg["use_lora"] = False
        cfg["full_finetune"] = False
        return "frozen"
    if mode == "lora":
        cfg["use_lora"] = True
        cfg["full_finetune"] = False
        return "lora"
    if mode == "full_finetune":
        cfg["use_lora"] = False
        cfg["full_finetune"] = True
        return "fullft"
    raise ValueError(f"Unsupported tuning mode '{tuning_mode}'")


def _build_training_cfg(base_cfg, variant: str, tuning_mode: str, seed: int, out_root: Path | None, epochs_override: int | None):
    cfg = deepcopy(base_cfg)
    cfg["seed"] = int(seed)
    if epochs_override is not None:
        cfg["epochs"] = int(epochs_override)

    dataset_type = str((cfg.get("dataset") or {}).get("type", "paired")).lower()
    if dataset_type != "paired":
        raise ValueError("This script expects dataset.type: paired")

    if int(cfg.get("batch_size", 2)) != 2:
        raise ValueError("Balanced paired training expects batch_size: 2")

    tuning_label = _apply_tuning_mode(cfg, tuning_mode)
    base_exp = str(cfg.get("experiment_id", "deepbacs_paired_balance"))
    task_type = str(cfg.get("task_type", "seg"))
    modality = str(cfg.get("modality", "deepbacs")).strip().lower() or "deepbacs"
    results_root = Path(out_root) if out_root is not None else Path(cfg.get("results_root", "results"))
    cfg["results_root"] = str(results_root)
    cfg["task_type"] = task_type
    cfg["modality"] = modality
    cfg["experiment_id"] = f"{base_exp}/{tuning_label}/{variant}/seed_{seed}"

    run_dir = results_root / modality / task_type / cfg["experiment_id"]
    cfg["out_dir"] = str(run_dir)
    return cfg


def _run_variant(variant, tuning_mode, seed, base_cfg, out_root: Path | None, epochs_override: int | None):
    cfg = _build_training_cfg(base_cfg, variant, tuning_mode, seed, out_root, epochs_override)
    run_dir = Path(cfg["out_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config_runtime.yaml"
    with cfg_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    trainer = SegTrainer(str(cfg_path))

    if variant == "balanced":
        sampling_cfg = _parse_balanced_sampling_cfg(cfg)
        group_a_idx, group_b_idx = _partition_indices_by_rules(trainer.train_ds.pairs, sampling_cfg)
        sampler = BalancedPairBatchSampler(group_a_idx, group_b_idx, seed=seed)
        trainer.train_loader = DataLoader(
            trainer.train_ds,
            batch_sampler=sampler,
            num_workers=cfg["num_workers"],
            pin_memory=(trainer.device.type == "cuda"),
            collate_fn=trainer._pad_collate,
        )

    trainer.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cfg",
        type=str,
        default="configs/cluster/deepbacs/deepbacs_paired_balanced_dinov2_cluster.yaml",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", type=str, choices=["unbalanced", "balanced", "both"], default="both")
    ap.add_argument(
        "--tuning-mode",
        type=str,
        choices=["cfg", "frozen", "lora", "full_finetune"],
        default="cfg",
        help="Override fine-tuning mode without duplicating configs.",
    )
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    base_cfg = _load_cfg(Path(args.cfg))
    out_root = Path(args.out_dir) if args.out_dir else None
    variants = ["unbalanced", "balanced"] if args.variant == "both" else [args.variant]
    for variant in variants:
        _run_variant(variant, args.tuning_mode, args.seed, base_cfg, out_root, args.epochs)


if __name__ == "__main__":
    main()
