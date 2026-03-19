from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image


def parse_center_crop_size(center_crop_size):
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


def center_crop_box(width: int, height: int, crop_w: int, crop_h: int):
    if crop_w > width or crop_h > height:
        raise ValueError(
            f"Requested center crop ({crop_w}, {crop_h}) is larger than image ({width}, {height})"
        )
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h
    return (left, top, right, bottom)


def open_first_frame(path) -> Image.Image:
    img = Image.open(path)
    try:
        if getattr(img, "n_frames", 1) > 1:
            img.seek(0)
    except Exception:
        pass
    return img


def _rescale_grayscale_to_uint8(arr: np.ndarray) -> np.ndarray:
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


def pil_grayscale_to_rgb(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[..., 0]

    if arr.dtype == np.uint8:
        gray = img.convert("L")
    else:
        gray = Image.fromarray(_rescale_grayscale_to_uint8(arr), mode="L")
    return Image.merge("RGB", (gray, gray, gray))


def load_rgb_image(path) -> Image.Image:
    img = open_first_frame(path)
    if img.mode == "RGB":
        return img
    return pil_grayscale_to_rgb(img)


def center_crop_image(image: Image.Image, center_crop_size) -> Image.Image:
    crop = parse_center_crop_size(center_crop_size)
    if crop is None:
        return image
    crop_h, crop_w = crop
    box = center_crop_box(
        width=image.width,
        height=image.height,
        crop_w=crop_w,
        crop_h=crop_h,
    )
    return image.crop(box)

