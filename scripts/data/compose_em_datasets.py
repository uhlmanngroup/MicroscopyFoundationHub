#!/usr/bin/env python3
"""Compose Drosophila + Lucchi++ into one dataset.

Writes:
  <OUT_ROOT>/{train,test}/{images,masks}/ and mapping.csv

Example (local; absolute paths are user-specific):
  python scripts/data/compose_em_datasets.py
"""

from pathlib import Path
import argparse
import csv, shutil, random

# =========================
# USER SETTINGS (edit here; absolute paths are user-specific)
# =========================
BASE = Path("/Users/cfuste/Documents/Data/ElectronMicroscopy")

# Drosophila stack1
DROSO_IMG = BASE / "groundtruth-drosophila-vnc-master/stack1/raw"
DROSO_MSK = BASE / "groundtruth-drosophila-vnc-master/stack1/mitochondria"
DROSO_TRAIN_RATIO = 0.85
SEED = 42

# Lucchi++ root (contains Train_In, Train_Out, Test_In, Test_Out)
LUCCHI_ROOT = BASE / "Lucchi++"

# Output dataset
OUT_ROOT = BASE / "composed-dinopeft"

# How to stage files: "symlink" (recommended) or "copy"
STAGE_MODE = "symlink"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# =========================
# helpers
# =========================
def list_images_recursive(d: Path):
    return sorted([p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])

def pair_by_stem(imgs, masks):
    m = {p.stem: p for p in masks}
    return [(ip, m[ip.stem]) for ip in imgs if ip.stem in m]

def pair_by_stem_strict(imgs, masks, label):
    if len({p.stem for p in imgs}) != len(imgs) or len({p.stem for p in masks}) != len(masks):
        raise RuntimeError(f"{label}: duplicate image or mask stems")
    pairs = pair_by_stem(imgs, masks)
    if len(pairs) != len(imgs) or len(pairs) != len(masks):
        raise RuntimeError(f"{label}: incomplete stem pairing: {len(imgs)} images, {len(masks)} masks, {len(pairs)} pairs")
    return pairs

def pair_by_sorted(imgs, masks):
    if len(imgs) != len(masks):
        raise RuntimeError(f"sorted pairing needs equal counts, got images={len(imgs)} masks={len(masks)}")
    return list(zip(imgs, masks))

def pair_lucchi(imgs, masks):
    """Match Lucchi's maskNNNN images to N masks by numeric slice ID."""
    def slice_id(path, prefix):
        stem = path.stem
        if prefix and stem.startswith(prefix):
            stem = stem[len(prefix):]
        if not stem.isdigit():
            raise ValueError(f"Unexpected Lucchi filename: {path.name}")
        return int(stem)

    mask_by_id = {}
    for mask in masks:
        key = slice_id(mask, "")
        if key in mask_by_id:
            raise ValueError(f"Duplicate Lucchi mask ID {key}: {mask_by_id[key]} and {mask}")
        mask_by_id[key] = mask

    pairs = []
    seen_images = set()
    for image in imgs:
        key = slice_id(image, "mask")
        if key in seen_images:
            raise ValueError(f"Duplicate Lucchi image ID {key}: {image}")
        seen_images.add(key)
        if key not in mask_by_id:
            raise ValueError(f"Missing Lucchi mask for {image.name} (ID {key})")
        pairs.append((image, mask_by_id[key]))

    extra_masks = set(mask_by_id) - seen_images
    if extra_masks:
        raise ValueError(f"Lucchi masks without images: {sorted(extra_masks)[:10]}")
    return pairs

def ensure_out_dirs(root: Path):
    for split in ("train","test"):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "masks").mkdir(parents=True, exist_ok=True)

def stage(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if STAGE_MODE == "symlink" and dst.resolve() != src.resolve():
            raise RuntimeError(
                f"Existing staged file points to the wrong source: {dst} -> {dst.resolve()}; "
                f"expected {src.resolve()}. Rebuild the composed dataset in a fresh directory."
            )
        return
    if STAGE_MODE == "copy":
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())

def stage_pairs(pairs, out_split_dir: Path, tag: str, split: str, start_idx: int, mapping_rows, counts):
    k = start_idx
    out_img = out_split_dir / "images"
    out_msk = out_split_dir / "masks"
    for ip, mp in pairs:
        new_stem = f"{tag}-{split}-{k:06d}"
        new_img = out_img / f"{new_stem}{ip.suffix.lower()}"
        new_msk = out_msk / f"{new_stem}{mp.suffix.lower()}"
        stage(ip, new_img)
        stage(mp, new_msk)
        mapping_rows.append([
            tag, split, new_stem,
            str(new_img.relative_to(OUT_ROOT)),
            str(new_msk.relative_to(OUT_ROOT)),
            str(ip.resolve()), str(mp.resolve())
        ])
        counts[(tag, split)] = counts.get((tag, split), 0) + 1
        k += 1
    return k

# =========================
# main
# =========================
def main():
    global LUCCHI_ROOT, OUT_ROOT, STAGE_MODE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lucchi-root", type=Path, default=LUCCHI_ROOT)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--stage-mode", choices=("symlink", "copy"), default=STAGE_MODE)
    ap.add_argument("--droso-train-images", type=Path)
    ap.add_argument("--droso-train-masks", type=Path)
    ap.add_argument("--droso-test-images", type=Path)
    ap.add_argument("--droso-test-masks", type=Path)
    args = ap.parse_args()
    LUCCHI_ROOT, OUT_ROOT, STAGE_MODE = args.lucchi_root, args.out_root, args.stage_mode
    split_dirs = (args.droso_train_images, args.droso_train_masks,
                  args.droso_test_images, args.droso_test_masks)
    if any(p is not None for p in split_dirs) and not all(p is not None for p in split_dirs):
        ap.error("Provide all four --droso-{train,test}-{images,masks} directories")

    # sanity paths
    for p in ([*split_dirs, LUCCHI_ROOT] if all(split_dirs) else [DROSO_IMG, DROSO_MSK, LUCCHI_ROOT]):
        if not p.exists():
            raise SystemExit(f"Missing path: {p}")

    if all(split_dirs) and OUT_ROOT.exists() and any(OUT_ROOT.iterdir()):
        raise SystemExit(f"Output directory is not empty: {OUT_ROOT}. Use a fresh --out-root.")

    ensure_out_dirs(OUT_ROOT)
    mapping_csv = OUT_ROOT / "mapping.csv"
    mapping_rows = [["dataset","split","new_stem","new_image_rel","new_mask_rel","src_image_abs","src_mask_abs"]]
    counts = {}

    # ---------- Drosophila (85/15 split) ----------
    if all(split_dirs):
        droso_train = pair_by_stem_strict(
            list_images_recursive(args.droso_train_images),
            list_images_recursive(args.droso_train_masks), "Droso train")
        droso_test = pair_by_stem_strict(
            list_images_recursive(args.droso_test_images),
            list_images_recursive(args.droso_test_masks), "Droso test")
        if len(droso_train) != 17 or len(droso_test) != 3:
            raise SystemExit(f"Expected VNC 17 train / 3 test, got {len(droso_train)} / {len(droso_test)}")
        overlap = {image.stem for image, _ in droso_train} & {image.stem for image, _ in droso_test}
        if overlap:
            raise SystemExit(f"VNC train/test slice overlap: {sorted(overlap)}")
    else:
        di = list_images_recursive(DROSO_IMG)
        dm = list_images_recursive(DROSO_MSK)
        if not di or not dm:
            raise SystemExit("Droso: no images or masks found (check paths/extensions).")
        droso_pairs = pair_by_stem(di, dm)
        if not droso_pairs:
            droso_pairs = pair_by_sorted(di, dm)
        random.Random(SEED).shuffle(droso_pairs)
        n_train = int(round(DROSO_TRAIN_RATIO * len(droso_pairs)))
        droso_train = droso_pairs[:n_train]
        droso_test = droso_pairs[n_train:]

    _ = stage_pairs(droso_train, OUT_ROOT / "train", "droso", "train", start_idx=0, mapping_rows=mapping_rows, counts=counts)
    _ = stage_pairs(droso_test,  OUT_ROOT / "test",  "droso", "test",  start_idx=0, mapping_rows=mapping_rows, counts=counts)

    # ---------- Lucchi++ (respect its Train/Test) ----------
    train_in  = LUCCHI_ROOT / "Train_In"
    train_out = LUCCHI_ROOT / "Train_Out"
    test_in   = LUCCHI_ROOT / "Test_In"
    test_out  = LUCCHI_ROOT / "Test_Out"
    for p in [train_in, train_out, test_in, test_out]:
        if not p.exists():
            raise SystemExit(f"Lucchi++ path missing: {p}")

    li_tr = list_images_recursive(train_in)
    lm_tr = list_images_recursive(train_out)
    li_te = list_images_recursive(test_in)
    lm_te = list_images_recursive(test_out)

    lucchi_train = pair_lucchi(li_tr, lm_tr)
    lucchi_test = pair_lucchi(li_te, lm_te)
    if len(lucchi_train) != 165 or len(lucchi_test) != 165:
        raise SystemExit(f"Expected Lucchi++ 165 train / 165 test, got {len(lucchi_train)} / {len(lucchi_test)}")
    _ = stage_pairs(lucchi_train, OUT_ROOT / "train", "lucchi", "train", start_idx=0, mapping_rows=mapping_rows, counts=counts)
    _ = stage_pairs(lucchi_test, OUT_ROOT / "test", "lucchi", "test", start_idx=0, mapping_rows=mapping_rows, counts=counts)

    # ---------- write mapping ----------
    with mapping_csv.open("w", newline="") as f:
        csv.writer(f).writerows(mapping_rows)

    # ---------- summary ----------
    train_imgs = len(list((OUT_ROOT / "train" / "images").iterdir()))
    test_imgs  = len(list((OUT_ROOT / "test" / "images").iterdir()))
    print("Composed dataset:", OUT_ROOT)
    print(f"  train images: {train_imgs} | droso={counts.get(('droso','train'),0)}  lucchi={counts.get(('lucchi','train'),0)}")
    print(f"  test  images: {test_imgs}  | droso={counts.get(('droso','test'),0)}   lucchi={counts.get(('lucchi','test'),0)}")
    print("Mapping CSV:", mapping_csv)
    if STAGE_MODE == "symlink":
        print("Note: symlinks created. If a tool can't follow symlinks, set STAGE_MODE='copy' and rerun.")

if __name__ == "__main__":
    main()
