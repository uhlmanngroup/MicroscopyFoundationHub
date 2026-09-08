#!/usr/bin/env python3
"""Download a random Open Images subset without pulling the full dataset.

Examples:
  python scripts/data/download_openimages_subset.py \
    --out-dir ~/data/openimages_500 \
    --count 500 \
    --seed 42

  python scripts/data/download_openimages_subset.py \
    --out-dir ~/data/openimages_500 \
    --count 500 \
    --seed 42 \
    --workers 12 \
    --overwrite
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SPLIT_TO_CSV_URL = {
    "validation": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
    "test": "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
}
S3_BASE_URL = "https://open-images-dataset.s3.amazonaws.com"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory for JPGs and metadata.")
    ap.add_argument("--count", type=int, default=500, help="Number of random images to download.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    ap.add_argument("--split", choices=tuple(SPLIT_TO_CSV_URL), default="validation")
    ap.add_argument("--workers", type=int, default=8, help="Parallel download workers.")
    ap.add_argument("--retries", type=int, default=3, help="Retries per image on transient failures.")
    ap.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds.")
    ap.add_argument("--overwrite", action="store_true", help="Redownload files that already exist.")
    return ap.parse_args()


def download_file(url: str, dst: Path, timeout: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        data = response.read()
    dst.write_bytes(data)


def ensure_csv(split: str, out_dir: Path, timeout: float, overwrite: bool) -> Path:
    csv_path = out_dir / f"{split}-images-with-rotation.csv"
    if csv_path.exists() and not overwrite:
        return csv_path
    url = SPLIT_TO_CSV_URL[split]
    print(f"[meta] downloading image list: {url}")
    download_file(url, csv_path, timeout=timeout)
    return csv_path


def load_image_ids(csv_path: Path) -> list[str]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        ids = [row["ImageID"] for row in reader if row.get("ImageID")]
    if not ids:
        raise RuntimeError(f"No ImageID rows found in {csv_path}")
    return ids


def choose_ids(image_ids: list[str], count: int, seed: int) -> list[str]:
    if count <= 0:
        raise ValueError("--count must be > 0")
    if count > len(image_ids):
        raise ValueError(f"Requested {count} images, but only found {len(image_ids)} IDs")
    rng = random.Random(seed)
    return sorted(rng.sample(image_ids, count))


def image_url(split: str, image_id: str) -> str:
    return f"{S3_BASE_URL}/{split}/{image_id}.jpg"


def download_one(
    split: str,
    image_id: str,
    dst_dir: Path,
    timeout: float,
    retries: int,
    overwrite: bool,
) -> tuple[str, bool, str | None]:
    dst = dst_dir / f"{image_id}.jpg"
    if dst.exists() and not overwrite:
        return image_id, False, None

    last_error: str | None = None
    for attempt in range(1, retries + 1):
        try:
            download_file(image_url(split, image_id), dst, timeout=timeout)
            return image_id, True, None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = f"attempt {attempt}/{retries}: {exc}"
            if attempt < retries:
                time.sleep(1.0)

    return image_id, False, last_error


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    image_list_path = out_dir / f"openimages_{args.split}_{args.count}_seed{args.seed}.txt"
    failures_path = out_dir / "failed_downloads.txt"

    csv_path = ensure_csv(args.split, out_dir, timeout=args.timeout, overwrite=False)
    image_ids = load_image_ids(csv_path)
    chosen_ids = choose_ids(image_ids, count=args.count, seed=args.seed)

    with image_list_path.open("w") as f:
        for image_id in chosen_ids:
            f.write(f"{args.split}/{image_id}\n")

    print(f"[sample] split={args.split} total_ids={len(image_ids)} chosen={len(chosen_ids)} seed={args.seed}")
    print(f"[sample] wrote subset list -> {image_list_path}")
    print(f"[download] target dir -> {images_dir}")

    failures: list[tuple[str, str]] = []
    downloaded = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                download_one,
                args.split,
                image_id,
                images_dir,
                args.timeout,
                args.retries,
                args.overwrite,
            ): image_id
            for image_id in chosen_ids
        }
        for idx, future in enumerate(as_completed(futures), start=1):
            image_id, was_downloaded, error = future.result()
            if error:
                failures.append((image_id, error))
                status = f"failed ({error})"
            elif was_downloaded:
                downloaded += 1
                status = "downloaded"
            else:
                skipped += 1
                status = "exists"
            print(f"[{idx:04d}/{len(chosen_ids):04d}] {image_id} -> {status}")

    if failures:
        with failures_path.open("w") as f:
            for image_id, error in failures:
                f.write(f"{args.split}/{image_id}\t{error}\n")
        print(f"[done] downloaded={downloaded} skipped={skipped} failed={len(failures)}")
        print(f"[done] failure log -> {failures_path}")
        sys.exit(1)

    print(f"[done] downloaded={downloaded} skipped={skipped} failed=0")
    print(f"[done] images dir -> {images_dir}")


if __name__ == "__main__":
    main()
