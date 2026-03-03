#!/usr/bin/env python3
"""
Tile paired image/mask directories into fixed-size patches (no resize).

Typical DeepBacs use:
    python scripts/utils/tile_paired_dirs.py \
        --src-images /path/to/single/train/images \
        --src-masks  /path/to/single/train/masks \
        --dst-images /path/to/single_512/train/images \
        --dst-masks  /path/to/single_512/train/masks \
        --patch-size 512 \
        --stride 512
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image


VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class PairItem:
    image: Path
    mask: Path


def _iter_files(root: Path, recursive: bool) -> List[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        p
        for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in VALID_EXTS and not p.name.startswith(".")
    )


def _pair_by_stem(images: Sequence[Path], masks: Sequence[Path]) -> List[PairItem]:
    mask_map: Dict[str, Path] = {}
    for m in masks:
        if m.stem in mask_map:
            raise RuntimeError(f"Duplicate mask stem detected: {m.stem}")
        mask_map[m.stem] = m
    pairs = [PairItem(image=i, mask=mask_map[i.stem]) for i in images if i.stem in mask_map]
    if not pairs:
        raise RuntimeError("No image/mask pairs found by stem.")
    return pairs


def _pair_by_sorted(images: Sequence[Path], masks: Sequence[Path]) -> List[PairItem]:
    if len(images) != len(masks):
        raise RuntimeError(
            f"Sorted pairing requires same counts, got images={len(images)} masks={len(masks)}"
        )
    return [PairItem(image=i, mask=m) for i, m in zip(images, masks)]


def _tile_starts(length: int, patch: int, stride: int, allow_partial: bool) -> List[int]:
    if length < patch:
        if allow_partial:
            return [0]
        return []
    starts = list(range(0, length - patch + 1, stride))
    if allow_partial:
        last = length - patch
        if starts[-1] != last:
            starts.append(last)
    return starts


def _save_crop(img: Image.Image, box: Tuple[int, int, int, int], out_path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop = img.crop(box)
    crop.save(out_path)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-images", type=Path, required=True)
    ap.add_argument("--src-masks", type=Path, required=True)
    ap.add_argument("--dst-images", type=Path, required=True)
    ap.add_argument("--dst-masks", type=Path, required=True)
    ap.add_argument("--patch-size", type=int, default=512, help="Square patch size.")
    ap.add_argument("--stride", type=int, default=512, help="Sliding-window stride.")
    ap.add_argument(
        "--pair-mode",
        choices=["stem", "sorted"],
        default="stem",
        help="How to pair image and mask files.",
    )
    ap.add_argument("--recursive", action="store_true", help="Recursively scan source directories.")
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Also emit border-aligned patches when image size is not divisible by stride/patch. "
            "If disabled, only full grid patches are generated."
        ),
    )
    ap.add_argument("--dry-run", action="store_true", help="Print stats only, no files written.")
    ap.add_argument("--overwrite", action="store_true", help="Allow overwriting destination patches.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    src_images = args.src_images.expanduser()
    src_masks = args.src_masks.expanduser()
    dst_images = args.dst_images.expanduser()
    dst_masks = args.dst_masks.expanduser()

    if args.patch_size <= 0:
        raise ValueError("--patch-size must be > 0")
    if args.stride <= 0:
        raise ValueError("--stride must be > 0")
    if not src_images.is_dir():
        raise FileNotFoundError(f"Source images dir not found: {src_images}")
    if not src_masks.is_dir():
        raise FileNotFoundError(f"Source masks dir not found: {src_masks}")

    image_files = _iter_files(src_images, recursive=bool(args.recursive))
    mask_files = _iter_files(src_masks, recursive=bool(args.recursive))
    if not image_files:
        raise RuntimeError(f"No valid image files found in {src_images}")
    if not mask_files:
        raise RuntimeError(f"No valid mask files found in {src_masks}")

    if args.pair_mode == "stem":
        pairs = _pair_by_stem(image_files, mask_files)
    else:
        pairs = _pair_by_sorted(image_files, mask_files)

    num_pairs = 0
    num_patches = 0
    skipped_small = 0
    skipped_nondisible = 0

    for item in pairs:
        with Image.open(item.image) as im, Image.open(item.mask) as mk:
            if im.size != mk.size:
                raise RuntimeError(
                    f"Size mismatch for pair '{item.image.name}' vs '{item.mask.name}': "
                    f"{im.size} vs {mk.size}"
                )
            w, h = im.size
            ys = _tile_starts(h, args.patch_size, args.stride, allow_partial=bool(args.allow_partial))
            xs = _tile_starts(w, args.patch_size, args.stride, allow_partial=bool(args.allow_partial))
            if not ys or not xs:
                skipped_small += 1
                continue
            if not args.allow_partial:
                if ((h - args.patch_size) % args.stride != 0) or ((w - args.patch_size) % args.stride != 0):
                    skipped_nondisible += 1
                    continue

            num_pairs += 1
            for y in ys:
                for x in xs:
                    box = (x, y, x + args.patch_size, y + args.patch_size)
                    out_stem = f"{item.image.stem}__y{y:04d}_x{x:04d}"
                    out_img = dst_images / f"{out_stem}{item.image.suffix.lower()}"
                    out_msk = dst_masks / f"{out_stem}{item.mask.suffix.lower()}"

                    if not args.overwrite and (out_img.exists() or out_msk.exists()):
                        raise FileExistsError(
                            f"Destination patch exists ({out_img.name} or {out_msk.name}). "
                            "Use --overwrite to replace."
                        )

                    _save_crop(im, box, out_img, dry_run=bool(args.dry_run))
                    _save_crop(mk, box, out_msk, dry_run=bool(args.dry_run))
                    num_patches += 1

    print(f"paired inputs discovered: {len(pairs)}")
    print(f"paired inputs processed: {num_pairs}")
    print(f"patches written: {num_patches}")
    print(f"skipped (too small): {skipped_small}")
    print(f"skipped (non-divisible grid, allow_partial=false): {skipped_nondisible}")
    if args.dry_run:
        print("dry-run mode: no files written")
    else:
        print(f"output images: {dst_images}")
        print(f"output masks : {dst_masks}")


if __name__ == "__main__":
    main()
