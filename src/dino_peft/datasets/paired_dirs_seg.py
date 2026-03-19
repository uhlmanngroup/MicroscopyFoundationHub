from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

from dino_peft.utils.image_size import (
    compute_resized_hw,
    parse_img_size_config,
)
from dino_peft.utils.image_loading import (
    center_crop_box as _center_crop_box,
    load_rgb_image,
    open_first_frame,
    parse_center_crop_size as _parse_center_crop_size,
)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def _list_files(root: Path, recursive: bool):
    pattern = "**/*" if recursive else "*"
    return sorted([p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in IMG_EXTS])

class PairedDirsSegDataset(Dataset):
    """
    Pair images and masks from two folders.

    Modes:
      - pair_mode="stem"   : match by filename stem (optionally remove mask_prefix/suffix)
      - pair_mode="sorted" : sort both lists and pair by index (requires same counts)

    Extras:
      - mask_prefix / mask_suffix for mask filenames like 'foo_mask.png' or 'gt_foo.png'
      - recursive=True to scan subfolders
      - binarize (0/255 → 0/1) or keep label ids

    Output:
      - image: Tensor CHW (after transform if provided)
      - mask : LongTensor HW (values in {0..K-1} or {0,1} if binarize)
    """
    def __init__(
        self,
        image_dir,
        mask_dir,
        img_size=(308, 308),
        to_rgb=True,
        transform=None,
        binarize=False,
        binarize_threshold=128,
        pair_mode="stem",
        mask_prefix="",
        mask_suffix="",
        recursive=False,
        center_crop_size=None,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir  = Path(mask_dir)
        self.transform = transform
        self.to_rgb    = bool(to_rgb)
        self.binarize  = bool(binarize)
        self.thresh    = int(binarize_threshold)
        self.resize_spec = parse_img_size_config(img_size)
        self.center_crop_size = _parse_center_crop_size(center_crop_size)

        imgs  = _list_files(self.image_dir, recursive)
        masks = _list_files(self.mask_dir,  recursive)

        self.pairs = []

        # Check that their are same number of images and masks
        if pair_mode == "sorted":
            if len(imgs) != len(masks):
                raise RuntimeError(
                    f"[pair_mode=sorted] Count mismatch: images={len(imgs)} masks={len(masks)}\n"
                    f"  images={self.image_dir}\n  masks={self.mask_dir}"
                )
            self.pairs = list(zip(imgs, masks))

        else:  # "stem" matching (with the name file)
            def norm_mask_stem(p: Path):
                s = p.stem
                if mask_prefix and s.startswith(mask_prefix):
                    s = s[len(mask_prefix):]
                if mask_suffix and s.endswith(mask_suffix):
                    s = s[: -len(mask_suffix)]
                return s

            mask_map = {norm_mask_stem(mp): mp for mp in masks}

            for ip in imgs:
                mp = mask_map.get(ip.stem, None)
                if mp is not None:
                    self.pairs.append((ip, mp))

        if len(self.pairs) == 0:
            peek_i = [p.name for p in imgs[:6]]
            peek_m = [p.name for p in masks[:6]]
            raise RuntimeError(
                "No (image,mask) pairs found.\n"
                f"  images={self.image_dir}\n"
                f"    first files: {peek_i}\n"
                f"  masks={self.mask_dir}\n"
                f"    first files: {peek_m}\n"
                "Try pair_mode='sorted' or set mask_prefix/mask_suffix/recursive."
            )

    def __len__(self):
        return len(self.pairs)

    def _load_rgb(self, p: Path):
        if self.to_rgb:
            return load_rgb_image(p)
        else:
            return open_first_frame(p)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]

        # Save in name the basename of the image path
        name = ip.stem
        # image
        img = self._load_rgb(ip)

        # --- MASK: force single channel ---
        mask = open_first_frame(mp)

        # Convert to 8-bit single-channel. This handles RGB/RGBA/Palette masks robustly.
        if mask.mode not in ("L", "I;16", "I"):
            try:
                mask = mask.convert("L")
            except Exception:
                # fallback: take first channel if convert fails (e.g., unusual mode)
                mask = mask.split()[0]

        # Optional aligned center crop for both image and mask.
        if self.center_crop_size is not None:
            crop_h, crop_w = self.center_crop_size
            if img.size != mask.size:
                raise ValueError(
                    "center_crop_size requires matching image/mask sizes before crop. "
                    f"Got image={img.size} mask={mask.size} for '{ip.name}'."
                )
            box = _center_crop_box(
                width=img.width,
                height=img.height,
                crop_w=crop_w,
                crop_h=crop_h,
            )
            img = img.crop(box)
            mask = mask.crop(box)

        target_hw = compute_resized_hw((img.height, img.width), self.resize_spec)
        target_wh = (target_hw[1], target_hw[0])
        if img.size != target_wh:
            img = img.resize(target_wh, Image.BICUBIC)
        mask = mask.resize(target_wh, Image.NEAREST)

        # transforms
        if self.transform:
            img = self.transform(img)

        # to numpy (single channel)
        mask = np.array(mask)
        if mask.ndim == 3:       # just in case something slipped through
            mask = mask[..., 0]

        # binarize or keep labels
        if self.binarize:
            mask = (mask > self.thresh).astype(np.int64)  # 0/1
        else:
            mask = mask.astype(np.int64)

        return img, torch.from_numpy(mask), name
