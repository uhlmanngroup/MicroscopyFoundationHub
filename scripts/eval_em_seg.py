#!/usr/bin/env python3
from pathlib import Path
import argparse, yaml, torch, numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dino_peft.datasets.paired_dirs_seg import PairedDirsSegDataset
from dino_peft.utils.transforms import em_seg_transforms
from dino_peft.models.backbone_dinov2 import DINOv2FeatureExtractor
from dino_peft.models.head_seg1x1 import SegHead1x1

# in scripts/eval_em_seg.py (replace eval_loop and add preview saving)

from torchvision.utils import save_image, make_grid
import torch.nn.functional as F

@torch.no_grad()
def colorize_mask(m: torch.Tensor, num_classes: int):
    B, H, W = m.shape
    out = torch.zeros(B, 3, H, W, device=m.device, dtype=torch.float32)
    if num_classes == 2:
        out[:, 0] = (m == 1).float()
        out[:, 1] = (m == 1).float()
        out[:, 2] = (m == 1).float()
    else:
        palette = torch.tensor([
            [0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],
            [1,0,1],[0,1,1],[1,0.5,0],[0.5,0,1],[0.5,0.5,0.5]
        ], device=m.device, dtype=torch.float32)
        for k in range(min(num_classes, palette.shape[0])):
            maskk = (m == k).unsqueeze(1).float()
            out += maskk * palette[k].view(1,3,1,1)
        out.clamp_(0,1)
    return out

@torch.no_grad()
def eval_loop(backbone, head, loader, device, num_classes, out_dir=None, preview_n=6):
    inter = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)
    tp = np.zeros(num_classes, dtype=np.float64)
    fp = np.zeros(num_classes, dtype=np.float64)
    fn = np.zeros(num_classes, dtype=np.float64)
    gt_pix = np.zeros(num_classes, dtype=np.float64)
    pr_pix = np.zeros(num_classes, dtype=np.float64)

    prev_count = 0
    prev_dir = None
    if out_dir is not None:
        prev_dir = Path(out_dir) / "eval_previews"
        prev_dir.mkdir(parents=True, exist_ok=True)

    for b, (imgs, masks) in enumerate(tqdm(loader, desc="eval")):
        imgs = imgs.to(device)
        masks = masks.to(device)
        logits = head(backbone(imgs), masks.shape[-2:])
        pred = logits.argmax(1)

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

        # save a few triptychs
        if prev_dir is not None and prev_count < preview_n:
            H, W = masks.shape[-2:]
            im = imgs.clamp(0,1)
            if im.shape[-2:] != (H,W):
                im = F.interpolate(im, size=(H,W), mode="bilinear", align_corners=False)
            pr_rgb = colorize_mask(pred, num_classes)
            gt_rgb = colorize_mask(masks, num_classes)
            trip = torch.cat([im, pr_rgb, gt_rgb], dim=0)
            save_image(make_grid(trip, nrow=im.shape[0]), prev_dir / f"sample_{b:04d}.png")
            prev_count += 1

    eps = 1e-7
    iou = inter / (union + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)

    print("Class pixel totals (GT):", gt_pix.astype(int).tolist())
    print("Class pixel totals (PR):", pr_pix.astype(int).tolist())
    for k in range(num_classes):
        if gt_pix[k] == 0:
            print(f"[WARN] test set has ZERO GT pixels for class {k} — IoU/Dice are not meaningful.")

    return iou, dice


# --- replace your `main()` with this version ---

def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _latest_ckpt(run_dir: Path) -> Path:
    cks = sorted(run_dir.glob("checkpoint_ep*.pt"))
    if not cks:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    return cks[-1]  # sorted by name epXXX; if you prefer mtime: max(cks, key=lambda p: p.stat().st_mtime)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Training YAML or saved config_used.yaml / config_runtime.yaml")
    ap.add_argument("--ckpt", default="", help="Optional checkpoint path; if empty, auto-pick latest in out_dir")
    ap.add_argument("--out_csv", default="", help="Optional metrics output; if empty, write out_dir/metrics_test.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    run_dir = Path(cfg["out_dir"]).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)

    # auto-pick checkpoint if not provided
    ckpt_path = Path(args.ckpt).expanduser() if args.ckpt else _latest_ckpt(run_dir)
    out_csv   = Path(args.out_csv).expanduser() if args.out_csv else (run_dir / "metrics_test.csv")

    device = _pick_device()

    # dataset (test split from cfg)
    t = em_seg_transforms(tuple(cfg["img_size"]))
    ds = PairedDirsSegDataset(
        cfg["val_img_dir"], cfg["val_mask_dir"],
        img_size=cfg["img_size"], to_rgb=True, transform=t,
        binarize=bool(cfg.get("binarize", True)),
        binarize_threshold=int(cfg.get("binarize_threshold", 128)),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4,
                        pin_memory=(device.type == "cuda"))

    # model
    bb = DINOv2FeatureExtractor(size=cfg["dino_size"], device=str(device))
    head = SegHead1x1(bb.embed_dim, cfg["num_classes"]).to(device)

    # load checkpoint (LoRA + head)
    ckpt = torch.load(ckpt_path, map_location=device)
    head.load_state_dict(ckpt["head"])
    bb_state = bb.state_dict()
    for k,v in ckpt.get("backbone_lora", {}).items():
        if k in bb_state:
            bb_state[k] = v
    bb.load_state_dict(bb_state, strict=False)

    bb.eval(); head.eval()
    iou, dice = eval_loop(bb, head, loader, device, cfg["num_classes"], out_dir=run_dir)

    # save CSV
    import csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class","IoU","Dice"])
        for k,(i,d) in enumerate(zip(iou, dice)):
            w.writerow([k, f"{i:.6f}", f"{d:.6f}"])
        w.writerow(["mean", f"{iou.mean():.6f}", f"{dice.mean():.6f}"])
    print(f"Saved metrics → {out_csv}")
    print(f"Used checkpoint → {ckpt_path}")


if __name__ == "__main__":
    main()
