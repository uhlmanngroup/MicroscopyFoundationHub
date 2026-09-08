#!/usr/bin/env python3
"""Prepare four binary segmentation datasets from MoNuSAC.

For each cell type (Epithelial, Lymphocyte, Macrophage, Neutrophil), produces:

    <out_root>/<cell_type>/
        train/
            images/   <stem>.tif   (copied from source)
            masks/    <stem>.png   (binary 0/255, foreground = that cell type)
        test/
            images/
            masks/

Patches whose XML has no polygons for a given cell type receive an all-zero mask
(pure background), so all four datasets always contain the same set of patches.
Cells annotated as "Ambiguous" are treated as background in all four datasets.
Patches missing a .tif counterpart are skipped with a warning.
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw


CELL_TYPES = ("Epithelial", "Lymphocyte", "Macrophage", "Neutrophil")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _collect_pairs(src_root: Path) -> list[tuple[Path, Path]]:
    """Return sorted (tif, xml) pairs found under src_root.

    Patches with no matching XML are skipped with a warning.
    """
    pairs: list[tuple[Path, Path]] = []
    for tif in sorted(src_root.rglob("*.tif")):
        xml = tif.with_suffix(".xml")
        if xml.is_file():
            pairs.append((tif, xml))
        else:
            print(f"  [skip] no XML for {tif.name}")
    return pairs


def _ensure_dirs(out_root: Path) -> None:
    for cell_type in CELL_TYPES:
        for split in ("train", "test"):
            for leaf in ("images", "masks"):
                (out_root / cell_type.lower() / split / leaf).mkdir(
                    parents=True, exist_ok=True
                )


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def _parse_polygons(xml_path: Path) -> dict[str, list[list[tuple[int, int]]]]:
    """Return {cell_type: [polygon, ...]} where polygon = [(x, y), ...]."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    polygons: dict[str, list[list[tuple[int, int]]]] = {ct: [] for ct in CELL_TYPES}

    for annotation in root.findall("Annotation"):
        attr = annotation.find("Attributes/Attribute")
        if attr is None:
            continue
        cell_type = attr.get("Name", "")
        if cell_type not in polygons:
            continue  # Ambiguous or unknown → background

        for region in annotation.findall("Regions/Region"):
            vertices = region.findall("Vertices/Vertex")
            if len(vertices) < 3:
                continue
            coords = [
                (int(float(v.get("X", 0))), int(float(v.get("Y", 0))))
                for v in vertices
            ]
            polygons[cell_type].append(coords)

    return polygons


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def _make_mask(
    size: tuple[int, int],
    polygon_list: list[list[tuple[int, int]]],
) -> Image.Image:
    """Rasterize polygons onto a binary L-mode mask of (width, height) = size."""
    mask = Image.new("L", size, 0)
    if polygon_list:
        draw = ImageDraw.Draw(mask)
        for coords in polygon_list:
            if len(coords) >= 3:
                draw.polygon(coords, fill=255)
    return mask


# ---------------------------------------------------------------------------
# Per-split processing
# ---------------------------------------------------------------------------

def _process_split(
    pairs: list[tuple[Path, Path]],
    split: str,
    out_root: Path,
) -> dict[str, int]:
    """Process one split (train or test). Returns {cell_type: patch_count}."""
    counts: dict[str, int] = {ct: 0 for ct in CELL_TYPES}

    for tif_path, xml_path in pairs:
        stem = tif_path.stem

        with Image.open(tif_path) as img:
            size = img.size  # (width, height)

        polygons = _parse_polygons(xml_path)

        for cell_type in CELL_TYPES:
            ct_dir = out_root / cell_type.lower() / split

            # Copy image (skip if already present from a previous run)
            dst_img = ct_dir / "images" / tif_path.name
            if not dst_img.exists():
                shutil.copy2(tif_path, dst_img)

            # Generate and save binary mask
            mask = _make_mask(size, polygons[cell_type])
            mask.save(ct_dir / "masks" / f"{stem}.png")

            counts[cell_type] += 1

    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--train-src",
        type=Path,
        required=True,
        help="MoNuSAC training folder (MoNuSAC_images_and_annotations)",
    )
    ap.add_argument(
        "--test-src",
        type=Path,
        required=True,
        help="MoNuSAC test folder (MoNuSAC Testing Data and Annotations)",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output root; cell-type subfolders are created here",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    train_src = args.train_src.expanduser().resolve()
    test_src = args.test_src.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()

    if not train_src.is_dir():
        raise FileNotFoundError(f"Training source not found: {train_src}")
    if not test_src.is_dir():
        raise FileNotFoundError(f"Test source not found: {test_src}")

    _ensure_dirs(out_root)

    print(f"Scanning training source: {train_src}")
    train_pairs = _collect_pairs(train_src)
    print(f"  {len(train_pairs)} patch(es) with matching XML\n")

    print(f"Scanning test source: {test_src}")
    test_pairs = _collect_pairs(test_src)
    print(f"  {len(test_pairs)} patch(es) with matching XML\n")

    print("Processing train split...")
    train_counts = _process_split(train_pairs, "train", out_root)
    print("Processing test split...")
    test_counts = _process_split(test_pairs, "test", out_root)

    print("\n=== Summary ===")
    print(f"{'Cell type':<15} {'Train':>6} {'Test':>6}")
    print("-" * 30)
    for ct in CELL_TYPES:
        print(f"{ct:<15} {train_counts[ct]:>6} {test_counts[ct]:>6}")
    print(f"\nOutput root: {out_root}")


if __name__ == "__main__":
    main()
