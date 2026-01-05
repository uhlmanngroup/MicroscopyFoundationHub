#!/usr/bin/env python3
"""Lightweight smoke test for the DINO backbone stack."""
from __future__ import annotations

"""
Dinov2 somke test:
$ python scripts/smoke_test.py \
  --dino-size small \
  --device cpu

Dinov3 smoke test:
$ python scripts/smoke_test.py \
  --backbone-name dinov3 \
  --variant vits16 \
  --weights /Users/cfuste/Documents/Models/DINOv3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --repo-dir /Users/cfuste/Documents/GitHub/dinov3 \
  --device cpu

"""

import argparse
import torch

from dino_peft.backbones import build_backbone, patch_tokens_to_grid, resolve_backbone_cfg
from dino_peft.models.head_seg1x1 import SegHeadDeconv


def resolve_device(device_str: str) -> torch.device:
    device_str = device_str.lower()
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a tiny forward pass through DINO backbone + seg head.")
    ap.add_argument("--backbone-name", default="dinov2", help="dinov2 or dinov3")
    ap.add_argument("--variant", default="small", help="Backbone variant (e.g., small, vits16).")
    ap.add_argument("--dino-size", default=None, help="Backward-compatible alias for --variant.")
    ap.add_argument("--weights", default=None, help="Optional weights path/URL (required for DINOv3).")
    ap.add_argument("--repo-dir", default=None, help="Optional local repo path for torch.hub.")
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda/mps")
    ap.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square image size (must be a multiple of the backbone patch size).",
    )
    ap.add_argument("--num-classes", type=int, default=2)
    args = ap.parse_args()

    device = resolve_device(args.device)
    img_size = int(args.image_size)
    variant = args.dino_size or args.variant
    backbone_cfg = resolve_backbone_cfg(
        {
            "backbone": {
                "name": args.backbone_name,
                "variant": variant,
                "weights": args.weights,
                "repo_dir": args.repo_dir,
                "load_backend": "torchhub",
            }
        }
    )
    if backbone_cfg["name"] == "dinov3" and not backbone_cfg.get("weights"):
        print("[smoke_test] DINOv3 weights not set; skipping DINOv3 smoke test.")
        return

    print(
        f"[smoke_test] device={device} backbone={backbone_cfg['name']}:{backbone_cfg['variant']} "
        f"img_size={img_size}"
    )

    backbone = build_backbone(backbone_cfg, device=str(device))
    if img_size % backbone.patch_size != 0:
        raise ValueError(
            f"--image-size must be a multiple of patch_size={backbone.patch_size}."
        )
    head = SegHeadDeconv(
        in_ch=backbone.embed_dim,
        num_classes=int(args.num_classes),
        n_ups=4,
        base_ch=128,
    ).to(device)

    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    output = backbone(dummy)
    feats = patch_tokens_to_grid(output)
    logits = head(feats, out_hw=(img_size, img_size))

    print(
        f"[smoke_test] feats={tuple(feats.shape)} logits={tuple(logits.shape)} "
        f"patch_tokens={tuple(output.patch_tokens.shape)} grid={output.grid_size}"
    )


if __name__ == "__main__":
    main()
