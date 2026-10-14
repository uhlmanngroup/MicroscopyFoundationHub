#!/usr/bin/env python3
"""Create a fresh EM triplet with Lucchi images and masks paired by slice ID.

The existing triplet is read only. Other datasets are linked from it; Lucchi
files are linked directly from the original Lucchi++ Train/Test folders.
"""

from __future__ import annotations

import argparse
from pathlib import Path


EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def image_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing directory: {folder}")
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS)


def lucchi_id(path: Path, prefix: str) -> int:
    stem = path.stem
    if prefix and stem.startswith(prefix):
        stem = stem[len(prefix):]
    if not stem.isdigit():
        raise ValueError(f"Unexpected Lucchi filename: {path.name}")
    return int(stem)


def originals_by_id(folder: Path, prefix: str) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in image_files(folder):
        key = lucchi_id(path, prefix)
        if key in found:
            raise ValueError(f"Duplicate Lucchi slice ID {key}: {found[key]} and {path}")
        found[key] = path
    return found


def plan_split(triplet_root: Path, lucchi_root: Path, split: str) -> list[tuple[Path, Path]]:
    source_images = image_files(triplet_root / split / "images")
    source_masks = image_files(triplet_root / split / "masks")
    masks_by_stem = {p.stem: p for p in source_masks}
    if len(masks_by_stem) != len(source_masks):
        raise ValueError(f"Duplicate mask stems in {split}")
    if {p.stem for p in source_images} != set(masks_by_stem):
        raise ValueError(f"Image/mask stems differ in triplet {split}")

    source_split = "Train" if split == "train" else "Test"
    lucchi_images = originals_by_id(lucchi_root / f"{source_split}_In", "mask")
    lucchi_masks = originals_by_id(lucchi_root / f"{source_split}_Out", "")
    if set(lucchi_images) != set(lucchi_masks):
        raise ValueError(f"Original Lucchi image/mask IDs differ in {source_split}")

    plan: list[tuple[Path, Path]] = []
    staged_lucchi_ids: set[int] = set()
    for staged_image in source_images:
        staged_mask = masks_by_stem[staged_image.stem]
        if staged_image.stem.startswith(f"lucchi-{split}-"):
            key = lucchi_id(staged_image, f"lucchi-{split}-")
            if key in staged_lucchi_ids:
                raise ValueError(f"Duplicate staged Lucchi ID {key} in {split}")
            staged_lucchi_ids.add(key)
            if key not in lucchi_images or key not in lucchi_masks:
                raise ValueError(f"Staged Lucchi ID {key} missing from original {source_split}")
            image_source = lucchi_images[key]
            mask_source = lucchi_masks[key]
        else:
            image_source = staged_image
            mask_source = staged_mask
        plan.append((image_source, Path(split) / "images" / staged_image.name))
        plan.append((mask_source, Path(split) / "masks" / staged_mask.name))

    if staged_lucchi_ids != set(lucchi_images):
        raise ValueError(f"Staged Lucchi IDs differ from original {source_split} IDs")
    print(f"{split}: {len(source_images)} image/mask pairs; {len(staged_lucchi_ids)} corrected Lucchi pairs")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplet-root", type=Path, required=True)
    parser.add_argument("--lucchi-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true", help="Validate sources without writing")
    args = parser.parse_args()

    triplet = args.triplet_root.expanduser().resolve()
    lucchi = args.lucchi_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    if output == triplet:
        raise ValueError("Output must be a new directory, separate from the original triplet")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    plan = plan_split(triplet, lucchi, "train") + plan_split(triplet, lucchi, "test")
    if args.check_only:
        print(f"Source validation passed; would create {len(plan)} links under {output}")
        return

    for source, relative in plan:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source.resolve())
    print(f"Created corrected triplet: {output}")


if __name__ == "__main__":
    main()
