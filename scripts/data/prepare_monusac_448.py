#!/usr/bin/env python3
"""Preprocess MoNuSAC patches to a uniform 448x448 size.

Three-tier logic applied to both image and mask:
  - min(W, H) < 200            -> drop (too small; resizing would distort cell scale too much)
  - 200 <= min(W, H) < 448     -> resize to 448x448 (bicubic image, nearest mask)
  - min(W, H) >= 448           -> center crop to 448x448

Output mirrors the input layout under a new root:
  <out_root>/<cell_type>/{train,test}/{images,masks}/<stem>.png

Usage:
    python scripts/data/prepare_monusac_448.py \
        --in-root /home/cfuste/data/datasets/monusac \
        --out-root /home/cfuste/data/datasets/monusac_448
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

TARGET = 448
MIN_KEEP = 200
CELL_TYPES = ("epithelial", "lymphocyte", "macrophage", "neutrophil")


def _center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def _process_split(
    in_root: Path, out_root: Path, cell_type: str, split: str
) -> dict[str, int]:
    in_img_dir = in_root / cell_type / split / "images"
    in_mask_dir = in_root / cell_type / split / "masks"
    out_img_dir = out_root / cell_type / split / "images"
    out_mask_dir = out_root / cell_type / split / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {"cropped": 0, "resized": 0, "dropped": 0, "no_mask": 0}

    for img_path in sorted(in_img_dir.iterdir()):
        if img_path.suffix.lower() not in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
            continue
        stem = img_path.stem
        mask_path = in_mask_dir / f"{stem}.png"
        if not mask_path.exists():
            print(f"    [skip] no mask for {img_path.name}")
            counts["no_mask"] += 1
            continue

        with Image.open(img_path) as _img:
            w, h = _img.size

        min_dim = min(w, h)

        if min_dim < MIN_KEEP:
            counts["dropped"] += 1
            continue

        with Image.open(img_path) as img:
            img = img.convert("RGB")
        with Image.open(mask_path) as mask:
            mask = mask.convert("L")

        if min_dim < TARGET:
            img_out = img.resize((TARGET, TARGET), Image.BICUBIC)
            mask_out = mask.resize((TARGET, TARGET), Image.NEAREST)
            counts["resized"] += 1
        else:
            img_out = _center_crop(img, TARGET)
            mask_out = _center_crop(mask, TARGET)
            counts["cropped"] += 1

        img_out.save(out_img_dir / f"{stem}.png")
        mask_out.save(out_mask_dir / f"{stem}.png")

    return counts


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--in-root", type=Path, required=True, help="Raw monusac root")
    ap.add_argument("--out-root", type=Path, required=True, help="Output root for processed data")
    args = ap.parse_args()

    in_root = args.in_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()

    print(f"Input:       {in_root}")
    print(f"Output:      {out_root}")
    print(f"Target size: {TARGET}x{TARGET}  |  Min keep: {MIN_KEEP}px\n")

    total_counts: dict[str, int] = {"cropped": 0, "resized": 0, "dropped": 0, "no_mask": 0}
    for cell_type in CELL_TYPES:
        for split in ("train", "test"):
            counts = _process_split(in_root, out_root, cell_type, split)
            kept = counts["cropped"] + counts["resized"]
            print(
                f"  {cell_type}/{split}: {kept} kept "
                f"(cropped={counts['cropped']}  resized={counts['resized']})  "
                f"dropped={counts['dropped']}"
                + (f"  no_mask={counts['no_mask']}" if counts["no_mask"] else "")
            )
            for k in total_counts:
                total_counts[k] += counts[k]

    total_kept = total_counts["cropped"] + total_counts["resized"]
    print(f"\nTotal: {total_kept} kept  "
          f"(cropped={total_counts['cropped']}  resized={total_counts['resized']})  "
          f"dropped={total_counts['dropped']}")
    print(f"\nOutput written to: {out_root}")


if __name__ == "__main__":
    main()
