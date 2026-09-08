#!/usr/bin/env python3
"""Stage flat image folders for domain-shift feature analysis.

The staged layout is:

  <out_root>/
    em/
      images/
      metadata.csv
    deepbacs/
      images/
      metadata.csv
    openimages/
      images/
      metadata.csv

Current defaults follow the agreed experiment setup:
- EM: use test images only from Lucchi++, Droso, Kasthuri++
- DeepBacs: use train + test images from coli, aureus, subtilis
- OpenImages: sample a flat local image folder down to `--openimages-count`

Examples:
  python scripts/data/stage_domain_shift_sets.py \
    --openimages-root /Users/cfuste/Documents/Data/openimages_500/images

  python scripts/data/stage_domain_shift_sets.py \
    --openimages-root /Users/cfuste/Documents/Data/openimages_500/images \
    --openimages-count 220 \
    --openimages-seed 42 \
    --out-root /Users/cfuste/Documents/Data/domain_shift
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
DEFAULT_OUT_ROOT = Path("/Users/cfuste/Documents/Data/domain_shift")
DEFAULT_EM_ROOT = Path("/Users/cfuste/Documents/Data/ElectronMicroscopy")
DEFAULT_DEEPBACS_ROOT = Path("/Users/cfuste/Documents/Data/DeepBacs/dino-poc/single")

METADATA_HEADER = [
    "domain",
    "source_dataset",
    "source_split",
    "sample_index",
    "staged_name",
    "staged_rel",
    "original_path_abs",
    "original_name",
    "original_stem",
]


@dataclass(frozen=True)
class Sample:
    domain: str
    source_dataset: str
    source_split: str
    path: Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--em-root", type=Path, default=DEFAULT_EM_ROOT)
    ap.add_argument("--deepbacs-root", type=Path, default=DEFAULT_DEEPBACS_ROOT)
    ap.add_argument(
        "--openimages-root",
        type=Path,
        required=True,
        help="Flat folder containing downloaded Open Images JPGs.",
    )
    ap.add_argument("--openimages-count", type=int, default=220)
    ap.add_argument("--openimages-seed", type=int, default=42)
    ap.add_argument(
        "--openimages-split",
        type=str,
        default="validation",
        help="Recorded metadata split for the local Open Images subset.",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def _iter_image_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")
    return sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS and not p.name.startswith(".")
    )


def _sanitize_token(text: str) -> str:
    token = text.strip().lower()
    token = re.sub(r"\s+", "_", token)
    token = re.sub(r"[^a-z0-9_+-]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "sample"


def _sanitize_stem(text: str) -> str:
    token = _sanitize_token(text)
    return token[:80] if len(token) > 80 else token


def _copy_file(src: Path, dst: Path, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not overwrite:
            return
        dst.unlink()
    shutil.copy2(src, dst)


def _stage_domain(
    out_root: Path,
    domain_name: str,
    samples: list[Sample],
    *,
    overwrite: bool,
    dry_run: bool,
) -> None:
    domain_root = out_root / domain_name
    images_root = domain_root / "images"
    metadata_path = domain_root / "metadata.csv"
    if not dry_run:
        domain_root.mkdir(parents=True, exist_ok=True)
        images_root.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[list[str]] = []
    per_source_counts: dict[tuple[str, str], int] = {}

    for idx, sample in enumerate(samples):
        source_key = (sample.source_dataset, sample.source_split)
        source_count = per_source_counts.get(source_key, 0)
        per_source_counts[source_key] = source_count + 1

        source_dataset = _sanitize_token(sample.source_dataset)
        source_split = _sanitize_token(sample.source_split)
        original_stem = _sanitize_stem(sample.path.stem)
        staged_name = (
            f"{source_dataset}-{source_split}-{source_count:06d}-{original_stem}{sample.path.suffix.lower()}"
        )
        staged_path = images_root / staged_name

        metadata_rows.append(
            [
                domain_name,
                sample.source_dataset,
                sample.source_split,
                str(idx),
                staged_name,
                staged_path.relative_to(domain_root).as_posix(),
                str(sample.path.resolve()),
                sample.path.name,
                sample.path.stem,
            ]
        )

        if not dry_run:
            _copy_file(sample.path, staged_path, overwrite=overwrite)

    if not dry_run:
        with metadata_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(METADATA_HEADER)
            writer.writerows(metadata_rows)

    print(f"[stage] domain={domain_name} samples={len(samples)} out={domain_root}")
    for (source_dataset, source_split), count in sorted(per_source_counts.items()):
        print(f"[stage]   {source_dataset} / {source_split}: {count}")
    if dry_run:
        print(f"[stage]   dry-run: metadata not written")
    else:
        print(f"[stage]   metadata -> {metadata_path}")


def collect_em_samples(em_root: Path) -> list[Sample]:
    mapping = {
        "lucchi": em_root / "Lucchi++" / "Test_In",
        "droso": em_root / "Droso" / "Test_In",
        "kasthuri": em_root / "Kasthuri++" / "Test_In",
    }
    samples: list[Sample] = []
    for dataset_name, root in mapping.items():
        for path in _iter_image_files(root):
            samples.append(
                Sample(
                    domain="em",
                    source_dataset=dataset_name,
                    source_split="test",
                    path=path,
                )
            )
    return samples


def collect_deepbacs_samples(deepbacs_root: Path) -> list[Sample]:
    dataset_roots = {
        "coli": deepbacs_root / "coli",
        "aureus": deepbacs_root / "aureus",
        "subtilis": deepbacs_root / "subtilis",
    }
    samples: list[Sample] = []
    for dataset_name, dataset_root in dataset_roots.items():
        for split in ("train", "test"):
            img_root = dataset_root / split / "images"
            for path in _iter_image_files(img_root):
                samples.append(
                    Sample(
                        domain="deepbacs",
                        source_dataset=dataset_name,
                        source_split=split,
                        path=path,
                    )
                )
    return samples


def collect_openimages_samples(
    openimages_root: Path,
    *,
    count: int,
    seed: int,
    split: str,
) -> list[Sample]:
    image_paths = _iter_image_files(openimages_root)
    if count <= 0:
        raise ValueError("--openimages-count must be > 0")
    if count > len(image_paths):
        raise ValueError(
            f"Requested {count} Open Images files, but only found {len(image_paths)} in {openimages_root}"
        )
    chosen = sorted(random.Random(seed).sample(image_paths, count))
    return [
        Sample(
            domain="openimages",
            source_dataset="openimages",
            source_split=split,
            path=path,
        )
        for path in chosen
    ]


def main() -> None:
    args = parse_args()
    out_root = args.out_root.expanduser().resolve()
    em_root = args.em_root.expanduser().resolve()
    deepbacs_root = args.deepbacs_root.expanduser().resolve()
    openimages_root = args.openimages_root.expanduser().resolve()

    em_samples = collect_em_samples(em_root)
    deepbacs_samples = collect_deepbacs_samples(deepbacs_root)
    openimages_samples = collect_openimages_samples(
        openimages_root,
        count=int(args.openimages_count),
        seed=int(args.openimages_seed),
        split=str(args.openimages_split),
    )

    print("[plan] staging domain-shift datasets")
    print(f"[plan] out_root       = {out_root}")
    print(f"[plan] em samples     = {len(em_samples)}")
    print(f"[plan] deepbacs       = {len(deepbacs_samples)}")
    print(f"[plan] openimages     = {len(openimages_samples)} (seed={args.openimages_seed})")

    _stage_domain(out_root, "em", em_samples, overwrite=args.overwrite, dry_run=args.dry_run)
    _stage_domain(
        out_root,
        "deepbacs",
        deepbacs_samples,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    _stage_domain(
        out_root,
        "openimages",
        openimages_samples,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
