#!/usr/bin/env python3
"""Optional instance-level postprocess evaluation on semantic checkpoints.

This script keeps the current semantic pipeline untouched and computes
object-level metrics after semantic inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import inspect
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from dino_peft.backbones import build_backbone, patch_tokens_to_grid, resolve_backbone_cfg
from dino_peft.datasets.droso_seg import DrosoSegDataset
from dino_peft.datasets.lucchi_seg import LucchiSegDataset
from dino_peft.datasets.paired_dirs_seg import PairedDirsSegDataset
from dino_peft.models.head_seg1x1 import SegHeadDeconv
from dino_peft.models.lora import apply_peft
from dino_peft.utils.image_size import DEFAULT_IMG_SIZE_CFG
from dino_peft.utils.paths import setup_run_dir, update_metrics
from dino_peft.utils.transforms import em_seg_transforms
from dino_peft.analysis.instance_postprocess import (
    aggregate_instance_metrics,
    evaluate_instance_image,
    label_connected_components,
    rows_to_dicts,
)


def _filter_dataset_params(dataset_class, dataset_params: dict, dataset_type: str) -> dict:
    sig = inspect.signature(dataset_class.__init__)
    allowed = {
        name
        for name, p in sig.parameters.items()
        if name != "self" and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    dropped = sorted(set(dataset_params) - allowed)
    if dropped:
        print(
            f"[postprocess_instance_eval] Ignoring unsupported dataset.params for type='{dataset_type}': {dropped}"
        )
    return {k: v for k, v in dataset_params.items() if k in allowed}


def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pad_collate(batch):
    imgs, masks, names = zip(*batch)
    max_h = max(img.shape[-2] for img in imgs)
    max_w = max(img.shape[-1] for img in imgs)
    padded_imgs = []
    padded_masks = []
    for img, mask in zip(imgs, masks):
        c = img.shape[0]
        h, w = img.shape[-2], img.shape[-1]
        pad_img = img.new_zeros((c, max_h, max_w))
        pad_img[:, :h, :w] = img
        padded_imgs.append(pad_img)

        mh, mw = mask.shape[-2], mask.shape[-1]
        pad_mask = mask.new_full((max_h, max_w), 0)
        pad_mask[:mh, :mw] = mask
        padded_masks.append(pad_mask)
    return torch.stack(padded_imgs), torch.stack(padded_masks), list(names)


def _resolve_modality(cfg: dict) -> str:
    modality = str(cfg.get("modality", "em")).strip().lower() or "em"
    cfg["modality"] = modality
    return modality


def _resolve_img_size(cfg: dict):
    modality = _resolve_modality(cfg)
    img_size_cfg = cfg.get("img_size")
    if modality == "deepbacs":
        requested_mode = None
        if isinstance(img_size_cfg, dict):
            requested_mode = str(img_size_cfg.get("mode", "")).lower()
        if requested_mode != "native":
            print(
                "[postprocess_instance_eval] modality=deepbacs forces img_size.mode='native' "
                "(no resizing in pipeline)."
            )
        img_size_cfg = {"mode": "native"}
        cfg["img_size"] = deepcopy(img_size_cfg)
        return img_size_cfg
    if img_size_cfg is None:
        img_size_cfg = deepcopy(DEFAULT_IMG_SIZE_CFG)
        cfg["img_size"] = deepcopy(img_size_cfg)
        print("[postprocess_instance_eval] img_size not set in config; defaulting to longest_edge=1022.")
    return img_size_cfg


def build_instance_dataset_from_cfg(cfg, split: str, transform):
    """
    Build dataset for postprocess eval with raw instance-id masks (no binarization).
    """
    img_size_cfg = _resolve_img_size(cfg)
    modality = _resolve_modality(cfg)
    dataset_cfg = cfg.get("dataset", {})
    dataset_type = str(dataset_cfg.get("type", "lucchi")).lower()
    dataset_params = dict(dataset_cfg.get("params", {}))
    dataset_map = {
        "lucchi": LucchiSegDataset,
        "droso": DrosoSegDataset,
        "paired": PairedDirsSegDataset,
    }
    DatasetClass = dataset_map.get(dataset_type)
    if DatasetClass is None:
        raise ValueError(
            f"Unsupported dataset.type '{dataset_type}'. "
            "Use 'lucchi', 'droso', or 'paired'."
        )
    if dataset_type == "lucchi":
        dataset_params.setdefault("recursive", False)
        dataset_params.setdefault("zfill_width", 4)
        dataset_params.setdefault("image_prefix", "mask")
    elif dataset_type == "droso":
        dataset_params.setdefault("recursive", True)
    if modality == "deepbacs":
        deepbacs_crop = int(cfg.get("deepbacs_center_crop_size", 448))
        if deepbacs_crop <= 0:
            raise ValueError(f"deepbacs_center_crop_size must be positive, got {deepbacs_crop}")
        dataset_params["center_crop_size"] = deepbacs_crop
    dataset_params = _filter_dataset_params(DatasetClass, dataset_params, dataset_type)

    kwargs = {
        "img_size": img_size_cfg,
        "to_rgb": True,
        "transform": transform,
        # Preserve raw instance ids for postprocess metrics.
        "binarize": False,
        "binarize_threshold": int(cfg.get("binarize_threshold", 128)),
    }
    kwargs.update(dataset_params)

    img_dir_key = f"{split}_img_dir"
    mask_dir_key = f"{split}_mask_dir"
    if img_dir_key not in cfg or mask_dir_key not in cfg:
        raise KeyError(f"Missing {img_dir_key}/{mask_dir_key} in config for split '{split}'")
    return DatasetClass(cfg[img_dir_key], cfg[mask_dir_key], **kwargs)


def best_checkpoint(run_dir) -> Path:
    p = Path(run_dir) / "ckpts" / "best_model.pt"
    if not p.exists():
        raise FileNotFoundError(f"best_model.pt not found in {p.parent}")
    return p


def _save_instance_mask(path: Path, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_label = int(labels.max()) if labels.size else 0
    dtype = np.uint16 if max_label <= np.iinfo(np.uint16).max else np.uint32
    img = Image.fromarray(labels.astype(dtype))
    img.save(path)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cfg",
        "--config",
        dest="cfg",
        required=True,
        help="Training YAML or saved config_used.yaml / config_runtime.yaml",
    )
    ap.add_argument("--ckpt", default="", help="Optional checkpoint path; if empty, use run_dir/ckpts/best_model.pt")
    ap.add_argument("--split", default="test", choices=["train", "test"], help="Dataset split for postprocess eval.")
    ap.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size.")
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    ap.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for instance matching.")
    ap.add_argument("--connectivity", type=int, default=8, choices=[4, 8], help="Connectivity for connected components.")
    ap.add_argument("--min-area", type=int, default=0, help="Drop predicted instances smaller than this area (pixels).")
    ap.add_argument("--save-instance-masks", action="store_true", help="Save postprocessed predicted instance label maps.")
    ap.add_argument(
        "--gt-binary-cc",
        action="store_true",
        help=(
            "Treat GT masks as binary foreground and convert them to instances via connected components "
            "(useful when GT is not already instance-labeled)."
        ),
    )
    ap.add_argument("--out-csv", default="", help="Per-image metrics CSV path. Default: run_dir/metrics_instance_postprocess.csv")
    ap.add_argument("--out-json", default="", help="Summary JSON path. Default: run_dir/metrics_instance_postprocess.json")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.cfg))

    task_type = cfg.get("task_type", "seg")
    if "experiment_id" in cfg and "results_root" in cfg:
        run_dir = setup_run_dir(
            cfg,
            task_type=task_type,
            subdirs=("figs", "instance_postprocess", "instance_postprocess/pred_instances"),
            save_config=False,
        )
    else:
        run_dir = Path(cfg["out_dir"]).expanduser()
        (run_dir / "instance_postprocess" / "pred_instances").mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.ckpt).expanduser() if args.ckpt else best_checkpoint(run_dir)
    if args.ckpt:
        run_dir = ckpt_path.parent.parent
        (run_dir / "instance_postprocess" / "pred_instances").mkdir(parents=True, exist_ok=True)

    out_csv = (
        Path(args.out_csv).expanduser()
        if args.out_csv
        else run_dir / "metrics_instance_postprocess.csv"
    )
    out_json = (
        Path(args.out_json).expanduser()
        if args.out_json
        else run_dir / "metrics_instance_postprocess.json"
    )

    print(f"[postprocess_instance_eval] run_dir={run_dir}")
    print(f"[postprocess_instance_eval] ckpt_path={ckpt_path}")

    device = _pick_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_cfg = ckpt.get("cfg", {}) or {}
    eval_cfg = ckpt_cfg if ckpt_cfg else cfg
    if ckpt_cfg:
        print("[postprocess_instance_eval] using cfg from checkpoint for model/data settings.")
    modality = _resolve_modality(eval_cfg)
    if modality != "deepbacs":
        print(
            "[postprocess_instance_eval][warn] modality is not deepbacs. "
            "Script still runs, but this workflow is intended for deepbacs instance-mask datasets."
        )

    backbone_cfg = resolve_backbone_cfg(eval_cfg)
    bb = build_backbone(backbone_cfg, device=device)
    apply_peft(
        bb.model,
        ckpt_cfg if ckpt_cfg else cfg,
        run_dir=run_dir,
        backbone_info=backbone_cfg,
        write_report=False,
    )
    bb.to(device)

    head = SegHeadDeconv(
        in_ch=bb.embed_dim,
        num_classes=eval_cfg["num_classes"],
        n_ups=4,
        base_ch=512,
    ).to(device)
    head.load_state_dict(ckpt["head"])

    backbone_full = ckpt.get("backbone")
    if backbone_full:
        bb.model.load_state_dict(backbone_full, strict=False)
    else:
        bb_state = bb.model.state_dict()
        lora_dict = ckpt.get("backbone_lora", {})
        for k, v in lora_dict.items():
            if k in bb_state:
                bb_state[k] = v
        bb.model.load_state_dict(bb_state, strict=False)

    bb.eval()
    head.eval()

    transform = em_seg_transforms(clahe_norm=bool(eval_cfg.get("clahe_norm", False)))
    ds = build_instance_dataset_from_cfg(eval_cfg, split=args.split, transform=transform)
    loader = DataLoader(
        ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=(device.type == "cuda"),
        collate_fn=pad_collate,
    )

    image_rows = []
    save_dir = run_dir / "instance_postprocess" / "pred_instances"
    global_idx = 0
    with torch.no_grad():
        for imgs, masks, names in tqdm(loader, desc="postprocess-eval"):
            imgs = imgs.to(device, non_blocking=True)
            out = bb(imgs)
            feats = patch_tokens_to_grid(out)
            logits = head(feats, masks.shape[-2:])
            pred_cls = logits.argmax(1).detach().cpu().numpy()
            gt_labels = masks.detach().cpu().numpy()

            for j in range(pred_cls.shape[0]):
                name = str(names[j])
                pred_binary = pred_cls[j] > 0
                gt_instance_labels = gt_labels[j]
                if args.gt_binary_cc:
                    gt_instance_labels = label_connected_components(
                        gt_instance_labels > 0,
                        connectivity=int(args.connectivity),
                        min_area=0,
                    )
                row, pred_instances = evaluate_instance_image(
                    name=name,
                    pred_binary=pred_binary,
                    gt_instance_labels=gt_instance_labels,
                    connectivity=int(args.connectivity),
                    min_area=int(args.min_area),
                    iou_threshold=float(args.iou_threshold),
                )
                image_rows.append(row)

                if args.save_instance_masks:
                    safe_name = name.replace("/", "_")
                    out_path = save_dir / f"{global_idx:06d}_{safe_name}_instances.tif"
                    _save_instance_mask(out_path, pred_instances)
                global_idx += 1

    summary = aggregate_instance_metrics(image_rows)
    summary["modality"] = modality
    summary["split"] = args.split
    summary["iou_threshold"] = float(args.iou_threshold)
    summary["connectivity"] = int(args.connectivity)
    summary["min_area"] = int(args.min_area)
    summary["gt_binary_cc"] = bool(args.gt_binary_cc)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_as_dict = rows_to_dicts(image_rows)
    fieldnames = [
        "name",
        "gt_instances",
        "pred_instances",
        "count_error",
        "abs_count_error",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "mean_matched_iou",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_as_dict)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    update_metrics(run_dir, "instance_postprocess", summary)

    print(f"[postprocess_instance_eval] per-image CSV → {out_csv}")
    print(f"[postprocess_instance_eval] summary JSON → {out_json}")
    print(
        "[postprocess_instance_eval] "
        f"count_mae={summary['count_mae']:.4f} "
        f"instance_f1={summary['instance_f1']:.4f} "
        f"mean_matched_iou={summary['mean_matched_iou']:.4f}"
    )


if __name__ == "__main__":
    main()
