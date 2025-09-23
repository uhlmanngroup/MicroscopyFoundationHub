#!/usr/bin/env python3
from pathlib import Path
import argparse, yaml, torch, numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dino_peft.datasets.paired_dirs_seg import PairedDirsSegDataset
from dino_peft.utils.transforms import em_seg_transforms
from dino_peft.models.backbone_dinov2 import DINOv2FeatureExtractor
from dino_peft.models.head_seg1x1 import SegHead1x1

@torch.no_grad()
def eval_loop(backbone, head, loader, device, num_classes):
    inter = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)
    tp = np.zeros(num_classes, dtype=np.float64)
    fp = np.zeros(num_classes, dtype=np.float64)
    fn = np.zeros(num_classes, dtype=np.float64)

    for imgs, masks in tqdm(loader, desc="eval"):
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

    eps = 1e-7
    iou = inter / (union + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    return iou, dice

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="same YAML used for training (for paths & num_classes)")
    ap.add_argument("--ckpt", required=True, help="checkpoint_epXXX.pt")
    ap.add_argument("--out_csv", required=True, help="where to save metrics csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset (use your composed TEST split)
    t = em_seg_transforms(tuple(cfg["img_size"]))
    ds = PairedDirsSegDataset(
        cfg["val_img_dir"], cfg["val_mask_dir"],
        img_size=cfg["img_size"], to_rgb=True, transform=t,
        binarize=bool(cfg.get("binarize", True)),
        binarize_threshold=int(cfg.get("binarize_threshold", 128)),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=(device.type=="cuda"))

    # model
    bb = DINOv2FeatureExtractor(size=cfg["dino_size"], device=str(device))
    head = SegHead1x1(bb.embed_dim, cfg["num_classes"]).to(device)

    # load checkpoint (LoRA weights + head)
    ckpt = torch.load(args.ckpt, map_location=device)
    # head
    head.load_state_dict(ckpt["head"])
    # LoRA: update only lora_* keys if present
    bb_state = bb.state_dict()
    for k,v in ckpt.get("backbone_lora", {}).items():
        if k in bb_state:
            bb_state[k] = v
    bb.load_state_dict(bb_state, strict=False)

    bb.eval(); head.eval()
    iou, dice = eval_loop(bb, head, loader, device, cfg["num_classes"])

    # save
    import csv
    out_p = Path(args.out_csv); out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class","IoU","Dice"])
        for k,(i,d) in enumerate(zip(iou, dice)):
            w.writerow([k, f"{i:.6f}", f"{d:.6f}"])
        w.writerow(["mean", f"{iou.mean():.6f}", f"{dice.mean():.6f}"])
    print("Saved metrics →", out_p)

if __name__ == "__main__":
    main()
