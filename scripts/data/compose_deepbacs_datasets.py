#!/usr/bin/env python3
"""Compose DeepBacs single datasets into paired and triple layouts.

The composed filenames preserve the source dataset name so balanced sampling can
recover group membership directly from the basename.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Sequence


VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PAIRED_COMBOS: tuple[tuple[str, ...], ...] = (
    ("coli", "subtilis"),
    ("coli", "aureus"),
    ("aureus", "subtilis"),
)
TRIPLE_COMBO: tuple[str, ...] = ("coli", "aureus", "subtilis")
CSV_HEADER = [
    "combo",
    "source_dataset",
    "split",
    "new_stem",
    "new_image_rel",
    "new_mask_rel",
    "src_image_abs",
    "src_mask_abs",
]


def _iter_files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.iterdir()
        if p.is_file()
        and p.suffix.lower() in VALID_EXTS
        and not p.name.startswith(".")
    )


def _pair_by_stem(images: Sequence[Path], masks: Sequence[Path]) -> list[tuple[Path, Path]]:
    mask_map = {m.stem: m for m in masks}
    pairs = [(img, mask_map[img.stem]) for img in images if img.stem in mask_map]
    if not pairs:
        raise RuntimeError("No image/mask pairs found by stem.")
    return pairs


def _ensure_layout(root: Path) -> None:
    for split in ("train", "test"):
        for leaf in ("images", "masks"):
            (root / split / leaf).mkdir(parents=True, exist_ok=True)


def _stage(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"Unsupported mode '{mode}'")


def _compose_combo(single_root: Path, out_root: Path, combo: Sequence[str], mode: str) -> None:
    combo_name = "-".join(combo)
    combo_root = out_root / combo_name
    _ensure_layout(combo_root)

    mapping_rows: list[list[str]] = []
    for split in ("train", "test"):
        next_index = 0
        for dataset_name in combo:
            img_root = single_root / dataset_name / split / "images"
            mask_root = single_root / dataset_name / split / "masks"
            if not img_root.is_dir():
                raise FileNotFoundError(f"Missing image dir: {img_root}")
            if not mask_root.is_dir():
                raise FileNotFoundError(f"Missing mask dir: {mask_root}")

            pairs = _pair_by_stem(_iter_files(img_root), _iter_files(mask_root))
            for img_path, mask_path in pairs:
                stem = f"{dataset_name}-{split}-{next_index:06d}"
                img_dst = combo_root / split / "images" / f"{stem}{img_path.suffix.lower()}"
                mask_dst = combo_root / split / "masks" / f"{stem}{mask_path.suffix.lower()}"
                _stage(img_path, img_dst, mode=mode)
                _stage(mask_path, mask_dst, mode=mode)
                mapping_rows.append(
                    [
                        combo_name,
                        dataset_name,
                        split,
                        stem,
                        img_dst.relative_to(combo_root).as_posix(),
                        mask_dst.relative_to(combo_root).as_posix(),
                        str(img_path.resolve()),
                        str(mask_path.resolve()),
                    ]
                )
                next_index += 1

    mapping_csv = combo_root / "mapping.csv"
    with mapping_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(mapping_rows)

    print(f"[done] {combo_root}")
    print(f"       mapping -> {mapping_csv}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--single-root", type=Path, required=True)
    ap.add_argument("--paired-root", type=Path, required=True)
    ap.add_argument("--triple-root", type=Path, required=True)
    ap.add_argument("--mode", choices=("copy", "symlink"), default="symlink")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    single_root = args.single_root.expanduser()
    paired_root = args.paired_root.expanduser()
    triple_root = args.triple_root.expanduser()

    if not single_root.is_dir():
        raise FileNotFoundError(f"single root not found: {single_root}")

    paired_root.mkdir(parents=True, exist_ok=True)
    triple_root.mkdir(parents=True, exist_ok=True)

    for combo in PAIRED_COMBOS:
        _compose_combo(single_root, paired_root, combo, mode=args.mode)
    _compose_combo(single_root, triple_root, TRIPLE_COMBO, mode=args.mode)


if __name__ == "__main__":
    main()
