#!/usr/bin/env python3
from pathlib import Path
import argparse, yaml, torch, numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import csv
import os
import mlflow
from mlflow import MlflowClient
import torch.nn.functional as F

from dino_peft.datasets.paired_dirs_seg import PairedDirsSegDataset
from dino_peft.datasets.lucchi_seg import LucchiSegDataset
from dino_peft.utils.transforms import em_seg_transforms, denorm_imagenet
from dino_peft.utils.viz import colorize_mask
from dino_peft.utils.plots import save_triptych_grid
from dino_peft.models.backbone_dinov2 import DINOv2FeatureExtractor
from dino_peft.models.head_seg1x1 import SegHeadDeconv
from dino_peft.models.lora import inject_lora
from dino_peft.utils.paths import setup_run_dir, update_metrics


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


def build_dataset_from_cfg(cfg, split: str, transform):
    dataset_cfg = cfg.get("dataset", {})
    dataset_type = str(dataset_cfg.get("type", "lucchi")).lower()
    dataset_params = dict(dataset_cfg.get("params", {}))
    dataset_map = {
        "lucchi": LucchiSegDataset,
        "paired": PairedDirsSegDataset,
    }
    DatasetClass = dataset_map.get(dataset_type)
    if DatasetClass is None:
        raise ValueError(
            f"Unsupported dataset.type '{dataset_type}'. "
            "Use 'lucchi' or 'paired'."
        )
    if dataset_type == "lucchi":
        dataset_params.setdefault("recursive", False)
        dataset_params.setdefault("zfill_width", 4)
        dataset_params.setdefault("image_prefix", "mask")

    kwargs = {
        "img_size": cfg["img_size"],
        "to_rgb": True,
        "transform": transform,
        "binarize": bool(cfg.get("binarize", True)),
        "binarize_threshold": int(cfg.get("binarize_threshold", 128)),
    }
    kwargs.update(dataset_params)

    img_dir_key = f"{split}_img_dir"
    mask_dir_key = f"{split}_mask_dir"
    if img_dir_key not in cfg or mask_dir_key not in cfg:
        raise KeyError(f"Missing {img_dir_key}/{mask_dir_key} in config for split '{split}'")

    return DatasetClass(cfg[img_dir_key], cfg[mask_dir_key], **kwargs)


@torch.no_grad()
def eval_loop(
    backbone, head, loader, device, num_classes,
    out_dir: Path | None = None, ckpt_tag: str = "model",
    preview_n: int = 4,              # kept for compatibility (unused when saving all)
    save_all_triptychs: bool = True 
):
    inter = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)
    tp = np.zeros(num_classes, dtype=np.float64)
    fp = np.zeros(num_classes, dtype=np.float64)
    fn = np.zeros(num_classes, dtype=np.float64)
    gt_pix = np.zeros(num_classes, dtype=np.float64)
    pr_pix = np.zeros(num_classes, dtype=np.float64)

    # Foreground-collapsed accumulators (bg=0, fg=1..K-1)
    fg_inter = 0.0
    fg_union = 0.0
    fg_tp = 0.0
    fg_fp = 0.0
    fg_fn = 0.0
    fg_gt_pix = 0.0
    fg_pr_pix = 0.0

    prev_dir = None
    if out_dir is not None:
        prev_dir = Path(out_dir) / "figs" / "eval_previews"
        prev_dir.mkdir(parents=True, exist_ok=True)

    for b, (imgs, masks, names) in enumerate(tqdm(loader, desc="eval")):
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        feats = backbone(imgs)
        logits = head(feats, masks.shape[-2:])
        pred = logits.argmax(1)

        # Per-class stats
        for k in range(num_classes):
            pk = (pred == k)
            mk = (masks == k)
            i = (pk & mk).sum().item()
            u = (pk | mk).sum().item()
            inter[k] += i
            union[k] += u
            tp[k] += i
            fp[k] += (pk & ~mk).sum().item()
            fn[k] += (~pk & mk).sum().item()
            gt_pix[k] += mk.sum().item()
            pr_pix[k] += pk.sum().item()

        # Foreground-collapsed (k>0)
        pk_fg = (pred > 0)
        mk_fg = (masks > 0)
        fg_inter += (pk_fg & mk_fg).sum().item()
        fg_union += (pk_fg | mk_fg).sum().item()
        fg_tp    += (pk_fg & mk_fg).sum().item()
        fg_fp    += (pk_fg & ~mk_fg).sum().item()
        fg_fn    += (~pk_fg & mk_fg).sum().item()
        fg_gt_pix += mk_fg.sum().item()
        fg_pr_pix += pk_fg.sum().item()
    
        # ---------------- Triptych saving (grouped) ----------------
        if prev_dir is not None:
            # de-normalize once per batch
            imgs_vis = denorm_imagenet(imgs.detach().cpu()).clamp(0, 1)   # (B,C,H?,W?)
            gts_cpu  = masks.detach().cpu()                               # (B,H,W)
            preds_cp = logits.detach().argmax(1).cpu()                    # (B,H,W)

            H, W = gts_cpu.shape[-2:]
            if imgs_vis.shape[-2:] != (H, W):
                imgs_vis = F.interpolate(imgs_vis, size=(H, W), mode="bilinear", align_corners=False)

            pred_rgb = colorize_mask(preds_cp, num_classes)               # (B,3,H,W)
            gt_rgb   = colorize_mask(gts_cpu,   num_classes)

            # ==== NEW: group into grids of N columns ====
            group_cols = 4                                  # <- change if you want
            if "trip_buf" not in locals():
                trip_buf = []                               # persistent across batches
                trip_count = 0

            B = imgs_vis.size(0)
            for j in range(B):
                trip_buf.append({
                    "image": imgs_vis[j],
                    "gt":    gt_rgb[j],
                    "pred":  pred_rgb[j],
                    "name":  str(names[j]),
                })
                # when buffer reaches N, flush to disk
                if len(trip_buf) == group_cols:
                    out_png = Path(prev_dir) / f"eval_triptych_{ckpt_tag}_g{trip_count:04d}.png"
                    save_triptych_grid(
                        trip_buf,
                        out_path=str(out_png),
                        title=f"Evaluation — {ckpt_tag} — group {trip_count}"
                    )
                    trip_buf = []
                    trip_count += 1

    eps = 1e-7
    iou  = inter / (union + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)

    iou_f  = fg_inter / (fg_union + eps)
    dice_f = (2 * fg_tp) / (2 * fg_tp + fg_fp + fg_fn + eps)

    # Sanity notes (optional)
    if fg_gt_pix == 0:
        print("[WARN] test set has ZERO foreground pixels — IoU_f/Dice_f not meaningful.")

    return iou, dice, iou_f, dice_f


# --- replace your `main()` with this version ---

def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _build_dataset(cfg, split: str, transform):
    split = split.lower()
    img_key = f"{split}_img_dir"
    mask_key = f"{split}_mask_dir"
    if img_key not in cfg or mask_key not in cfg:
        raise KeyError(f"Config missing '{img_key}' or '{mask_key}' for eval split='{split}'")

    dataset_cfg = cfg.get("dataset", {})
    dataset_type = str(dataset_cfg.get("type", "lucchi")).lower()
    params = dataset_cfg.get("params") or {}
    common = dict(
        image_dir=cfg[img_key],
        mask_dir=cfg[mask_key],
        img_size=tuple(cfg["img_size"]),
        to_rgb=True,
        transform=transform,
        binarize=bool(cfg.get("binarize", True)),
        binarize_threshold=int(cfg.get("binarize_threshold", 128)),
    )
    if dataset_type == "paired":
        return PairedDirsSegDataset(
            **common,
            pair_mode=params.get("pair_mode", "stem"),
            mask_prefix=params.get("mask_prefix", ""),
            mask_suffix=params.get("mask_suffix", ""),
            recursive=bool(params.get("recursive", False)),
        )
    elif dataset_type == "lucchi":
        return LucchiSegDataset(
            **common,
            recursive=bool(params.get("recursive", False)),
            zfill_width=int(params.get("zfill_width", 4)),
            image_prefix=params.get("image_prefix", "mask"),
        )
    else:
        raise ValueError(
            f"Unknown dataset.type '{dataset_type}'. "
            "Use 'lucchi' or 'paired' and attach details under dataset.params."
        )

def best_checkpoint(run_dir) -> Path:
    """Return ckpts/best_model.pt or raise if missing."""
    p = Path(run_dir) / "ckpts" / "best_model.pt"
    if not p.exists():
        raise FileNotFoundError(f"best_model.pt not found in {p.parent}")
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Training YAML or saved config_used.yaml / config_runtime.yaml")
    ap.add_argument("--ckpt", default="", help="Optional checkpoint path; if empty, auto-pick latest in run directory")
    ap.add_argument("--out_csv", default="", help="Optional metrics output; if empty, write run_dir/metrics_test.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    task_type = cfg.get("task_type", "seg")
    if "experiment_id" in cfg and "results_root" in cfg:
        run_dir = setup_run_dir(
            cfg,
            task_type=task_type,
            subdirs=("figs", "figs/eval_previews"),
            save_config=False,
        )
    else:
        run_dir = Path(cfg["out_dir"]).expanduser()
        (run_dir / "figs" / "eval_previews").mkdir(parents=True, exist_ok=True)

    # auto-pick checkpoint if not provided
    ckpt_path = Path(args.ckpt).expanduser() if args.ckpt else best_checkpoint(run_dir)
    out_csv   = Path(args.out_csv).expanduser() if args.out_csv else (run_dir / "metrics_test.csv")

    device = _pick_device()

    # dataset (test split from cfg)
    t = em_seg_transforms()
    ds = build_dataset_from_cfg(cfg, split="test", transform=t)

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        collate_fn=pad_collate,
    )

    # model
    bb = DINOv2FeatureExtractor(size=cfg["dino_size"], device=str(device))
    use_lora = cfg.get("use_lora", True)
    if use_lora:
        targets = cfg.get("lora_targets", ["attn.qkv", "attn.proj"])
        r = int(cfg.get("lora_rank", 8))
        alpha = int(cfg.get("lora_alpha", 16))
        lora_names = inject_lora(bb.vit, target_substrings=targets, r=r, alpha=alpha)

    bb.to(device)
    head = SegHeadDeconv(in_ch=bb.embed_dim, num_classes=cfg["num_classes"], n_ups=4, base_ch=512).to(device)

    # load checkpoint (LoRA + head)
    ckpt = torch.load(ckpt_path, map_location=device)
    head.load_state_dict(ckpt["head"])
    bb_state = bb.state_dict()
    lora_dict = ckpt.get("backbone_lora", {})
    matched = 0
    for k, v in lora_dict.items():
        if k in bb_state:
            bb_state[k] = v
            matched += 1
    bb.load_state_dict(bb_state, strict=False)

    bb.eval(); head.eval()
    # --- compute metrics ---
    iou, dice, iou_f, dice_f = eval_loop(bb, head, loader, device, cfg["num_classes"], out_dir=run_dir)

    # --- write CSV BEFORE logging to MLflow ---
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class","IoU","Dice"])
        for k,(i,d) in enumerate(zip(iou, dice)):
            w.writerow([k, f"{i:.6f}", f"{d:.6f}"])
        w.writerow(["foreground", f"{iou_f:.6f}", f"{dice_f:.6f}"])
        w.writerow(["mean", f"{iou.mean():.6f}", f"{dice.mean():.6f}"])
    print(f"Saved metrics → {out_csv}")
    update_metrics(
        run_dir,
        "eval",
        {
            "mean_iou": float(iou.mean()),
            "mean_dice": float(dice.mean()),
            "foreground_iou": float(iou_f),
            "foreground_dice": float(dice_f),
            "num_classes": int(cfg["num_classes"]),
        },
    )

    # --- single MLflow run ---
    mlflow.set_tracking_uri(f"file:{(Path.cwd()/'mlruns').as_posix()}")
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "default"))

    with mlflow.start_run(run_name="eval") as run:
        # force hydration and print where it’s going
        mlflow.log_param("phase", "eval")
        mlflow.log_metric("canary/step0", 0.0, step=0)
        print("[mlflow:eval] cwd=", Path.cwd().as_posix(),
              "tracking_uri=", mlflow.get_tracking_uri(),
              "run_id=", run.info.run_id,
              "artifact_uri=", mlflow.get_artifact_uri())

        # metrics
        mlflow.log_metric("test/iou_f",  float(iou_f))
        mlflow.log_metric("test/Dice_fg", float(dice_f))

        # artifacts
        mlflow.log_artifact(str(out_csv), artifact_path="eval")
        prev_dir = run_dir / "figs" / "eval_previews"
        if prev_dir.exists():
            mlflow.log_artifacts(str(prev_dir), artifact_path="eval_previews")



if __name__ == "__main__":
    main()
