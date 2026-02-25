#!/usr/bin/env python3
"""
Compose Kasthuri++ into an existing TRIPLET-style dataset layout.

It reads:
    Kasthuri++/Train_In   -> train/images
    Kasthuri++/Train_Out  -> train/masks
    Kasthuri++/Test_In    -> test/images
    Kasthuri++/Test_Out   -> test/masks

and renames files as:
    <dataset-name>-<split>-<zero-padded-index>.<ext>

The index is reassigned sequentially from 0 within each split (train/test),
based on the numeric portion of the original filename (e.g. mask1031.png).

A CSV mapping is written with the same columns used in the user's existing
mapping.csv example.

Examples:
    # Preview only
    python scripts/utils/compose_triplet_kasthuri.py --dry-run

    # Copy files and write a new mapping CSV
    python scripts/utils/compose_triplet_kasthuri.py \
        --mode copy \
        --mapping-csv /Users/cfuste/Documents/Data/ElectronMicroscopy/TRIPLET/mapping_kasthuri.csv

    # Move files (destructive to source) and append rows to an existing mapping
    python scripts/utils/compose_triplet_kasthuri.py \
        --mode move \
        --append \
        --mapping-csv /Users/cfuste/Documents/Data/ElectronMicroscopy/composed-dinopeft-PAIRED/mapping.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


DEFAULT_KASTHURI_ROOT = Path("/Users/cfuste/Documents/Data/ElectronMicroscopy/Kasthuri++")
DEFAULT_TRIPLET_ROOT = Path("/Users/cfuste/Documents/Data/ElectronMicroscopy/TRIPLET")
CSV_HEADER = [
    "dataset",
    "split",
    "new_stem",
    "new_image_rel",
    "new_mask_rel",
    "src_image_abs",
    "src_mask_abs",
]
SPLIT_DIRS = {
    "train": ("Train_In", "Train_Out"),
    "test": ("Test_In", "Test_Out"),
}


@dataclass(frozen=True)
class FilePairTask:
    dataset: str
    split: str
    index: int
    src_image: Path
    src_mask: Path
    dst_image: Path
    dst_mask: Path
    triplet_root: Path

    @property
    def new_stem(self) -> str:
        return f"{self.dataset}-{self.split}-{self.index:06d}"

    def mapping_row(self) -> list[str]:
        return [
            self.dataset,
            self.split,
            self.new_stem,
            self.dst_image.relative_to(self.triplet_root).as_posix(),
            self.dst_mask.relative_to(self.triplet_root).as_posix(),
            str(self.src_image.resolve()),
            str(self.src_mask.resolve()),
        ]


def parse_numeric_key(filename: str) -> tuple[int, str]:
    """Extract the last integer in a filename stem for ordering."""
    stem = Path(filename).stem
    matches = re.findall(r"(\d+)", stem)
    if not matches:
        raise ValueError(f"Could not parse numeric slice id from filename: {filename}")
    return int(matches[-1]), filename


def visible_files(folder: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        files[p.name] = p
    return files


def build_split_tasks(kasthuri_root: Path, triplet_root: Path, dataset_name: str, split: str) -> list[FilePairTask]:
    in_dir_name, out_dir_name = SPLIT_DIRS[split]
    src_images_dir = kasthuri_root / in_dir_name
    src_masks_dir = kasthuri_root / out_dir_name
    dst_images_dir = triplet_root / split / "images"
    dst_masks_dir = triplet_root / split / "masks"

    if not src_images_dir.is_dir():
        raise FileNotFoundError(f"Missing source images directory: {src_images_dir}")
    if not src_masks_dir.is_dir():
        raise FileNotFoundError(f"Missing source masks directory: {src_masks_dir}")

    img_files = visible_files(src_images_dir)
    mask_files = visible_files(src_masks_dir)

    img_names = set(img_files)
    mask_names = set(mask_files)
    if img_names != mask_names:
        missing_in_masks = sorted(img_names - mask_names)
        missing_in_images = sorted(mask_names - img_names)
        raise ValueError(
            "Image/mask filename mismatch in split "
            f"{split}. missing_in_masks={missing_in_masks[:10]} "
            f"missing_in_images={missing_in_images[:10]}"
        )

    ordered_names = sorted(img_names, key=parse_numeric_key)
    tasks: list[FilePairTask] = []
    for new_idx, name in enumerate(ordered_names):
        src_img = img_files[name]
        src_mask = mask_files[name]
        stem = f"{dataset_name}-{split}-{new_idx:06d}"
        dst_img = dst_images_dir / f"{stem}{src_img.suffix}"
        dst_mask = dst_masks_dir / f"{stem}{src_mask.suffix}"
        tasks.append(
            FilePairTask(
                dataset=dataset_name,
                split=split,
                index=new_idx,
                src_image=src_img,
                src_mask=src_mask,
                dst_image=dst_img,
                dst_mask=dst_mask,
                triplet_root=triplet_root,
            )
        )
    return tasks


def validate_destinations(tasks: list[FilePairTask], overwrite: bool) -> None:
    seen: set[Path] = set()
    for task in tasks:
        for dst in (task.dst_image, task.dst_mask):
            if dst in seen:
                raise ValueError(f"Duplicate destination generated: {dst}")
            seen.add(dst)
            if dst.exists() and not overwrite:
                raise FileExistsError(
                    f"Destination already exists: {dst}\n"
                    "Use --overwrite if you intentionally want to replace it."
                )


def ensure_mapping_header(mapping_csv: Path, append: bool) -> None:
    if append and mapping_csv.exists():
        with mapping_csv.open("r", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                header = []
        if header and header != CSV_HEADER:
            raise ValueError(
                f"Existing CSV header does not match expected format.\n"
                f"Expected: {CSV_HEADER}\nFound:    {header}"
            )


def execute_tasks(
    tasks: list[FilePairTask],
    mapping_csv: Path,
    mode: str,
    append: bool,
    overwrite: bool,
    dry_run: bool,
) -> None:
    if dry_run:
        print(f"[dry-run] Planned pairs: {len(tasks)}")
        for task in tasks[:5]:
            print(f"[dry-run] {task.src_image.name} -> {task.dst_image}")
            print(f"[dry-run] {task.src_mask.name}  -> {task.dst_mask}")
        if len(tasks) > 5:
            print(f"[dry-run] ... ({len(tasks) - 5} more pairs)")
        print(f"[dry-run] Mapping CSV would be written to: {mapping_csv}")
        return

    for task in tasks:
        task.dst_image.parent.mkdir(parents=True, exist_ok=True)
        task.dst_mask.parent.mkdir(parents=True, exist_ok=True)

    ensure_mapping_header(mapping_csv, append=append)
    mapping_csv.parent.mkdir(parents=True, exist_ok=True)

    move_fn = shutil.move if mode == "move" else shutil.copy2

    rows = [task.mapping_row() for task in tasks]
    for task in tasks:
        for src, dst in ((task.src_image, task.dst_image), (task.src_mask, task.dst_mask)):
            if overwrite and dst.exists():
                dst.unlink()
            move_fn(str(src), str(dst))
        print(f"[{mode}] {task.src_image.name} -> {task.dst_image.name}")

    write_mode = "a" if append and mapping_csv.exists() else "w"
    should_write_header = write_mode == "w" or mapping_csv.stat().st_size == 0
    with mapping_csv.open(write_mode, newline="") as f:
        writer = csv.writer(f)
        if should_write_header:
            writer.writerow(CSV_HEADER)
        writer.writerows(rows)

    completed_verb = "moved" if mode == "move" else "copied"
    print(f"Done. {completed_verb} {len(tasks)} image/mask pairs.")
    print(f"Mapping CSV: {mapping_csv}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kasthuri-root", type=Path, default=DEFAULT_KASTHURI_ROOT)
    ap.add_argument("--triplet-root", type=Path, default=DEFAULT_TRIPLET_ROOT)
    ap.add_argument(
        "--dataset-name",
        default="kasthuri",
        help="Prefix used in renamed files, e.g. kasthuri-train-000000.png",
    )
    ap.add_argument(
        "--mapping-csv",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <triplet-root>/mapping_<dataset-name>.csv",
    )
    ap.add_argument(
        "--mode",
        choices=("copy", "move"),
        default="copy",
        help="copy (safe default) or move (removes files from Kasthuri++ source)",
    )
    ap.add_argument("--append", action="store_true", help="Append rows to an existing mapping CSV")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite destination files if they exist")
    ap.add_argument("--dry-run", action="store_true", help="Validate and print planned operations only")
    args = ap.parse_args()

    kasthuri_root = args.kasthuri_root.expanduser()
    triplet_root = args.triplet_root.expanduser()
    dataset_name = args.dataset_name.strip().lower()
    if not dataset_name:
        raise ValueError("--dataset-name cannot be empty")

    mapping_csv = (
        args.mapping_csv.expanduser()
        if args.mapping_csv is not None
        else triplet_root / f"mapping_{dataset_name}.csv"
    )

    all_tasks: list[FilePairTask] = []
    for split in ("train", "test"):
        split_tasks = build_split_tasks(kasthuri_root, triplet_root, dataset_name, split)
        all_tasks.extend(split_tasks)
        print(f"[plan] {split}: {len(split_tasks)} pairs")

    validate_destinations(all_tasks, overwrite=args.overwrite)
    execute_tasks(
        all_tasks,
        mapping_csv=mapping_csv,
        mode=args.mode,
        append=args.append,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
