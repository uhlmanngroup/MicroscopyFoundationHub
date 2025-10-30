# src/dino_peft/datasets/lucchi_seg.py
from pathlib import Path
from typing import Iterable, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# Reuse the robust I/O and tensor logic from your generic dataset
from .paired_dirs_seg import PairedDirsSegDataset

def _list_images(root: Path, recursive: bool) -> list[Path]:
    pats = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
    if recursive:
        out = []
        for p in pats:
            out.extend(root.rglob(p))
        return sorted(out)
    else:
        out = []
        for p in pats:
            out.extend(root.glob(p))
        return sorted(out)

class LucchiSegDataset(PairedDirsSegDataset):
    """
    Lucchi-specific pairing:
      images: mask0000.png, mask0001.png, ...
      masks : 0.png, 1.png, 2.png, ...
    Strategy:
      strip 'mask' prefix from IMAGE stems and zfill both stems to width=4,
      then pair by normalized stem.
    Everything else (loading, resizing, binarize, transform) is inherited.
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
        recursive=False,
        zfill_width: int = 4,
        image_prefix: str = "mask",   # default for Lucchi
    ):
        # We don't call super().__init__ because that would run its generic pairing.
        # Instead, we set the attributes it expects, then reuse its __getitem__ / _load_rgb.
        self.image_dir = Path(image_dir)
        self.mask_dir  = Path(mask_dir)
        self.img_size  = tuple(img_size)
        self.to_rgb    = bool(to_rgb)
        self.transform = transform
        self.binarize  = bool(binarize)
        self.thresh    = int(binarize_threshold)

        imgs  = _list_images(self.image_dir, recursive)
        masks = _list_images(self.mask_dir,  recursive)

        # Build normalization helpers
        def norm_img_stem(p: Path) -> str:
            s = p.stem
            if image_prefix and s.startswith(image_prefix):
                s = s[len(image_prefix):]     # "mask0007" -> "0007"
            # if it's pure number after stripping, zfill; else keep as-is
            return s.zfill(zfill_width) if s.isdigit() else s

        def norm_mask_stem(p: Path) -> str:
            s = p.stem                     # "7" -> "0007"
            return s.zfill(zfill_width) if s.isdigit() else s

        mask_map = {norm_mask_stem(mp): mp for mp in masks}

        pairs: list[Tuple[Path, Path]] = []
        missed = []

        for ip in imgs:
            key = norm_img_stem(ip)
            mp = mask_map.get(key, None)
            if mp is not None:
                pairs.append((ip, mp))
            else:
                missed.append((ip.name, key))

        self.pairs = pairs

        if len(self.pairs) == 0:
            peek_i = [p.name for p in imgs[:6]]
            peek_m = [p.name for p in masks[:6]]
            raise RuntimeError(
                "LucchiSegDataset: No (image,mask) pairs found.\n"
                f"  images={self.image_dir}\n"
                f"    first files: {peek_i}\n"
                f"  masks={self.mask_dir}\n"
                f"    first files: {peek_m}\n"
                f"  missed examples (image_name -> norm_key): {missed[:6]}"
            )

    # __len__, __getitem__, and _load_rgb are inherited from PairedDirsSegDataset
