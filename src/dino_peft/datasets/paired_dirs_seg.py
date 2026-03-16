from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

from dino_peft.utils.image_size import (
    compute_resized_hw,
    parse_img_size_config,
)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _parse_center_crop_size(center_crop_size):
    if center_crop_size is None:
        return None
    if isinstance(center_crop_size, int):
        if center_crop_size <= 0:
            raise ValueError(f"center_crop_size must be positive, got {center_crop_size}")
        return (center_crop_size, center_crop_size)
    if isinstance(center_crop_size, (tuple, list)):
        if len(center_crop_size) != 2:
            raise ValueError(
                f"center_crop_size tuple/list must have len 2, got {center_crop_size}"
            )
        h, w = int(center_crop_size[0]), int(center_crop_size[1])
        if h <= 0 or w <= 0:
            raise ValueError(f"center_crop_size values must be positive, got {center_crop_size}")
        return (h, w)
    raise TypeError(
        "center_crop_size must be None, int, or tuple/list of length 2 "
        f"(got {type(center_crop_size)})"
    )


def _center_crop_box(width: int, height: int, crop_w: int, crop_h: int):
    if crop_w > width or crop_h > height:
        raise ValueError(
            f"Requested center crop ({crop_w}, {crop_h}) is larger than image ({width}, {height})"
        )
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h
    return (left, top, right, bottom)

def _list_files(root: Path, recursive: bool):
    pattern = "**/*" if recursive else "*"
    return sorted([p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in IMG_EXTS])


def _rescale_grayscale_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Robustly map a single grayscale image to uint8 using per-image percentiles.

    This is only used for high-bit-depth microscopy images. Standard uint8 inputs
    keep the legacy path unchanged.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grayscale array, got shape {arr.shape}")

    lo = float(np.quantile(arr, 0.001))
    hi = float(np.quantile(arr, 0.999))
    if hi <= lo:
        lo = float(np.min(arr))
        hi = float(np.max(arr))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return np.round(arr * 255.0).astype(np.uint8)


def _pil_grayscale_to_rgb(img: Image.Image) -> Image.Image:
    """
    Preserve contrast for high-bit-depth grayscale images while keeping the
    existing uint8 path unchanged for EM datasets.
    """
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[..., 0]

    if arr.dtype == np.uint8:
        gray = img.convert("L")
    else:
        gray = Image.fromarray(_rescale_grayscale_to_uint8(arr), mode="L")
    return Image.merge("RGB", (gray, gray, gray))

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
        img = Image.open(p)
        # if TIFF has multiple frames, take the first
        try:
            if getattr(img, "n_frames", 1) > 1:
                img.seek(0)
        except Exception:
            pass
        if self.to_rgb:
            if img.mode == "RGB":
                return img
            # Typical EM is uint8 grayscale; some DeepBacs TIFFs are 16-bit and
            # need per-image rescaling before converting to RGB.
            return _pil_grayscale_to_rgb(img)
        else:
            return img

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]

        # Save in name the basename of the image path
        name = ip.stem
        # image
        img = self._load_rgb(ip)

        # --- MASK: force single channel ---
        mask = Image.open(mp)
        try:
            if getattr(mask, "n_frames", 1) > 1:
                mask.seek(0)
        except Exception:
            pass

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
