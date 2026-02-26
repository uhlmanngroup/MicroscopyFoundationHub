#!/usr/bin/env python3
"""
Center-crop TRIPLET images into a new folder for feature analysis.

This utility creates a new TRIPLET-like tree containing only image folders:
    <dst_root>/train/images
    <dst_root>/test/images

It preserves filenames and file extensions (e.g. .png, .tif).

Example:
    python scripts/utils/center_crop_triplet_images.py \
        --src-root /Users/cfuste/Documents/Data/ElectronMicroscopy/TRIPLET \
        --dst-root /Users/cfuste/Documents/Data/ElectronMicroscopy/TRIPLET_C112 \
        --size 112
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


DEFAULT_SRC = Path("/Users/cfuste/Documents/Data/ElectronMicroscopy/TRIPLET")
DEFAULT_DST = Path("/Users/cfuste/Documents/Data/ElectronMicroscopy/TRIPLET_C224")
VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def center_crop_box(width: int, height: int, crop_w: int, crop_h: int) -> tuple[int, int, int, int]:
    if crop_w > width or crop_h > height:
        raise ValueError(
            f"Crop {crop_w}x{crop_h} is larger than image {width}x{height}."
        )
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h
    return left, top, right, bottom


def iter_images(folder: Path):
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() not in VALID_EXTS:
            continue
        yield p


def crop_split(src_dir: Path, dst_dir: Path, crop_w: int, crop_h: int, overwrite: bool, dry_run: bool) -> int:
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source image folder not found: {src_dir}")
    files = list(iter_images(src_dir))
    if not files:
        raise ValueError(f"No valid images found in {src_dir}")

    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for src_path in files:
        dst_path = dst_dir / src_path.name
        if dst_path.exists() and not overwrite:
            raise FileExistsError(f"Destination exists: {dst_path} (use --overwrite)")

        with Image.open(src_path) as img:
            box = center_crop_box(img.width, img.height, crop_w, crop_h)
            cropped = img.crop(box)

            if dry_run:
                if count < 5:
                    print(f"[dry-run] {src_path.name}: {img.width}x{img.height} -> {crop_w}x{crop_h}")
                count += 1
                continue

            # Re-save with same extension/format inferred from destination.
            cropped.save(dst_path)
            count += 1

    return count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-root", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--dst-root", type=Path, default=DEFAULT_DST)
    ap.add_argument(
        "--size",
        type=int,
        default=224,
        help="Center crop size (square), e.g. 112 for 112x112.",
    )
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Splits to process (default: train test).",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing cropped images.")
    ap.add_argument("--dry-run", action="store_true", help="Validate and preview without writing files.")
    args = ap.parse_args()

    src_root = args.src_root.expanduser()
    dst_root = args.dst_root.expanduser()
    crop_size = int(args.size)
    if crop_size <= 0:
        raise ValueError("--size must be positive")

    total = 0
    for split in args.splits:
        split = str(split).strip().lower()
        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported split '{split}'. Expected train/test.")
        src_dir = src_root / split / "images"
        dst_dir = dst_root / split / "images"
        n = crop_split(
            src_dir=src_dir,
            dst_dir=dst_dir,
            crop_w=crop_size,
            crop_h=crop_size,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        total += n
        print(f"[{split}] processed {n} images -> {dst_dir}")

    print(f"Done. Total processed images: {total}")


if __name__ == "__main__":
    main()
