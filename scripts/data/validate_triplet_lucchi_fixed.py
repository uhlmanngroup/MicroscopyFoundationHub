#!/usr/bin/env python3
"""Validate every Lucchi slice in a corrected EM triplet and its mapping CSV."""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

from rebuild_triplet_lucchi import EXTS, lucchi_id, originals_by_id


HEADER = [
    "dataset", "split", "new_stem", "new_image_rel", "new_mask_rel",
    "src_image_abs", "src_mask_abs",
]
EXPECTED_GROUPS = {"droso", "kasthuri", "lucchi"}


def files_by_stem(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing directory: {folder}")
    files = [p for p in folder.iterdir() if p.suffix.lower() in EXTS]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing or broken data file: {path}")
    by_stem = {p.stem: p for p in files}
    if len(by_stem) != len(files):
        raise ValueError(f"Duplicate image stems in {folder}")
    return by_stem


def validate(dataset_root: Path, lucchi_root: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for split, source_split in (("train", "Train"), ("test", "Test")):
        images = files_by_stem(dataset_root / split / "images")
        masks = files_by_stem(dataset_root / split / "masks")
        if images.keys() != masks.keys():
            raise ValueError(f"Image/mask stems differ in {split}")
        original_images = originals_by_id(lucchi_root / f"{source_split}_In", "mask")
        original_masks = originals_by_id(lucchi_root / f"{source_split}_Out", "")
        if original_images.keys() != original_masks.keys():
            raise ValueError(f"Original Lucchi IDs differ in {source_split}")

        lucchi_ids: set[int] = set()
        groups: set[str] = set()
        for stem in sorted(images):
            image, mask = images[stem], masks[stem]
            dataset = stem.split("-", 1)[0]
            if dataset not in EXPECTED_GROUPS or not stem.startswith(f"{dataset}-{split}-"):
                raise ValueError(f"Unexpected staged sample: {stem}")
            groups.add(dataset)
            image_source = image.resolve(strict=True)
            mask_source = mask.resolve(strict=True)
            if dataset == "lucchi":
                slice_id = lucchi_id(image, f"lucchi-{split}-")
                if slice_id in lucchi_ids:
                    raise ValueError(f"Duplicate Lucchi ID {slice_id} in {split}")
                lucchi_ids.add(slice_id)
                if slice_id not in original_images:
                    raise ValueError(f"Lucchi ID {slice_id} missing in original {source_split}")
                expected_image = original_images[slice_id].resolve(strict=True)
                expected_mask = original_masks[slice_id].resolve(strict=True)
                if image_source != expected_image or mask_source != expected_mask:
                    raise ValueError(
                        f"Incorrect Lucchi pair {stem}: {image_source}, {mask_source}; "
                        f"expected {expected_image}, {expected_mask}"
                    )
            rows.append([
                dataset, split, stem,
                str(image.relative_to(dataset_root)),
                str(mask.relative_to(dataset_root)),
                str(image_source), str(mask_source),
            ])

        if groups != EXPECTED_GROUPS:
            raise ValueError(f"Expected {sorted(EXPECTED_GROUPS)} in {split}, found {sorted(groups)}")
        if lucchi_ids != set(original_images):
            raise ValueError(f"Staged Lucchi IDs differ from original {source_split} IDs")
        print(f"{split}: {len(images)} valid pairs, {len(lucchi_ids)} correctly paired Lucchi slices")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--lucchi-root", type=Path, required=True)
    parser.add_argument("--write-mapping", action="store_true",
                        help="Create or repair mapping.csv after validating all pairs")
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve(strict=True)
    lucchi = args.lucchi_root.expanduser().resolve(strict=True)
    rows = validate(root, lucchi)
    mapping = root / "mapping.csv"

    if args.write_mapping:
        with tempfile.NamedTemporaryFile("w", newline="", dir=root,
                                         prefix=".mapping-", suffix=".csv",
                                         delete=False) as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            writer.writerows(rows)
            temporary = Path(handle.name)
        temporary.replace(mapping)
        print(f"Wrote {mapping}")

    if not mapping.is_file():
        raise FileNotFoundError(f"Missing {mapping}; rerun with --write-mapping")
    with mapping.open(newline="") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != HEADER or list(reader) != rows:
            raise ValueError(f"Mapping CSV differs from the staged files: {mapping}")
    print(f"Validated {len(rows)} mapping rows in {mapping}")


if __name__ == "__main__":
    main()
