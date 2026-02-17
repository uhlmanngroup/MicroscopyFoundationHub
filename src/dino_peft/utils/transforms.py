from torchvision import transforms as T
import torch 
from PIL import Image
import numpy as np
import random

from .image_size import compute_resized_hw, parse_img_size_config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
OPENCLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENCLIP_STD = [0.26862954, 0.26130258, 0.27577711]

def _clahe_gray_uint8(gray: np.ndarray, clip_limit: float, tiles: tuple[int, int]) -> np.ndarray:
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("CLAHE expects a uint8 grayscale image.")
    h, w = gray.shape
    ty, tx = tiles
    if h == 0 or w == 0:
        return gray
    y_edges = np.linspace(0, h, ty + 1, dtype=int)
    x_edges = np.linspace(0, w, tx + 1, dtype=int)
    out = np.empty_like(gray)
    bins = 256

    for yi in range(ty):
        y0, y1 = int(y_edges[yi]), int(y_edges[yi + 1])
        if y1 <= y0:
            continue
        for xi in range(tx):
            x0, x1 = int(x_edges[xi]), int(x_edges[xi + 1])
            if x1 <= x0:
                continue
            tile = gray[y0:y1, x0:x1]
            hist = np.bincount(tile.reshape(-1), minlength=bins).astype(np.int64)

            tile_area = int(tile.size)
            clip_th = max(1, int(round(clip_limit * tile_area / bins)))
            excess = np.maximum(hist - clip_th, 0)
            hist = np.minimum(hist, clip_th)
            n_excess = int(excess.sum())
            if n_excess > 0:
                base = n_excess // bins
                rem = n_excess % bins
                hist += base
                if rem > 0:
                    hist[:rem] += 1

            cdf = hist.cumsum()
            nz = np.flatnonzero(cdf)
            if nz.size == 0:
                out[y0:y1, x0:x1] = tile
                continue
            cdf_min = int(cdf[nz[0]])
            denom = int(tile_area - cdf_min)
            if denom <= 0:
                lut = np.arange(256, dtype=np.uint8)
            else:
                lut = np.clip(((cdf - cdf_min) * 255.0) / denom, 0, 255).astype(np.uint8)
            out[y0:y1, x0:x1] = lut[tile]
    return out


class _EMSubtleCLAHE:
    """
    Subtle CLAHE preprocessing for EM-like images.
    - Works on grayscale luminance.
    - Blends CLAHE output with the original to keep the effect mild.
    """

    def __init__(
        self,
        clip_limit: float = 1.5,
        tiles: tuple[int, int] = (8, 8),
        blend_alpha: float = 0.35,
    ):
        self.clip_limit = float(clip_limit)
        self.tiles = (int(tiles[0]), int(tiles[1]))
        self.blend_alpha = float(blend_alpha)

    def __call__(self, image: Image.Image) -> Image.Image:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
        clahe = _clahe_gray_uint8(gray, clip_limit=self.clip_limit, tiles=self.tiles)
        alpha = min(max(self.blend_alpha, 0.0), 1.0)
        mixed = np.round((1.0 - alpha) * gray + alpha * clahe).astype(np.uint8)
        if image.mode == "RGB":
            return Image.merge("RGB", (Image.fromarray(mixed), Image.fromarray(mixed), Image.fromarray(mixed)))
        return Image.fromarray(mixed)


def em_seg_transforms(img_size=(308,308), clahe_norm: bool = False):
    ops = []
    if clahe_norm:
        ops.append(_EMSubtleCLAHE())
    ops.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return T.Compose(ops)


def _flip_array(arr: np.ndarray, axis_mode: str) -> np.ndarray:
    if axis_mode in ("h", "hv"):
        arr = np.flip(arr, axis=1)
    if axis_mode in ("v", "hv"):
        arr = np.flip(arr, axis=0)
    return np.ascontiguousarray(arr)


def _shift_array_reflect(arr: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if dx == 0 and dy == 0:
        return arr
    h, w = arr.shape[:2]
    pad_x = abs(dx)
    pad_y = abs(dy)
    pad_spec = [(pad_y, pad_y), (pad_x, pad_x)]
    if arr.ndim == 3:
        pad_spec.append((0, 0))
    pad_mode = "reflect"
    if (h == 1 and pad_y > 0) or (w == 1 and pad_x > 0):
        pad_mode = "edge"
    padded = np.pad(arr, pad_spec, mode=pad_mode)
    y0 = pad_y - dy
    x0 = pad_x - dx
    return np.ascontiguousarray(padded[y0:y0 + h, x0:x0 + w])


def _shift_mask_zero(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if dx == 0 and dy == 0:
        return mask
    h, w = mask.shape[:2]
    out = np.zeros_like(mask)
    span_w = w - abs(dx)
    span_h = h - abs(dy)
    if span_w <= 0 or span_h <= 0:
        return out
    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    dst_x0 = max(0, dx)
    dst_y0 = max(0, dy)
    src_x1 = src_x0 + span_w
    src_y1 = src_y0 + span_h
    dst_x1 = dst_x0 + span_w
    dst_y1 = dst_y0 + span_h
    out[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return out


class EMSegFlipShiftAug:
    """
    Joint EM segmentation augmentation applied to image+mask:
      1) Always flip using a random axis mode from {h, v, hv}
      2) Always shift by up to max_shift_percent (per axis)
         - image border uses reflect padding
         - mask border is filled with class 0
    """

    def __init__(self, max_shift_percent: float = 0.05):
        if max_shift_percent < 0 or max_shift_percent >= 1:
            raise ValueError("max_shift_percent must be in [0, 1).")
        self.max_shift_percent = float(max_shift_percent)

    def __call__(self, image: Image.Image, mask: Image.Image):
        img_np = np.asarray(image)
        mask_np = np.asarray(mask)

        axis_mode = random.choice(("h", "v", "hv"))
        img_np = _flip_array(img_np, axis_mode)
        mask_np = _flip_array(mask_np, axis_mode)

        h, w = img_np.shape[:2]
        max_dx = int(round(w * self.max_shift_percent))
        max_dy = int(round(h * self.max_shift_percent))
        dx = random.randint(-max_dx, max_dx) if max_dx > 0 else 0
        dy = random.randint(-max_dy, max_dy) if max_dy > 0 else 0

        img_np = _shift_array_reflect(img_np, dx=dx, dy=dy)
        mask_np = _shift_mask_zero(mask_np, dx=dx, dy=dy)

        return Image.fromarray(img_np), Image.fromarray(mask_np)


def em_seg_joint_augment(enabled: bool, max_shift_percent: float = 0.05):
    if not enabled:
        return None
    return EMSegFlipShiftAug(max_shift_percent=max_shift_percent)

def denorm_imagenet(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, C, H, W) normalized with ImageNet stats.
    Returns a de-normalized tensor in [approximately 0..1] range (clamp later).
    """
    if x.dim() != 4:
        raise ValueError(f"Expected (B,C,H,W), got {tuple(x.shape)}")
    if x.size(1) != 3:
        # nothing to do for non-RGB; return as-is
        return x
    mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std  = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return x * std + mean

# Helper is declared at module scope so it remains picklable for DataLoader workers.
class _ResizeLongestEdge:
    def __init__(self, resize_spec):
        self.resize_spec = resize_spec

    def __call__(self, image: Image.Image) -> Image.Image:
        target_hw = compute_resized_hw((image.height, image.width), self.resize_spec)
        target_wh = (target_hw[1], target_hw[0])
        if image.size != target_wh:
            return image.resize(target_wh, Image.BICUBIC)
        return image


def em_dino_unsup_transforms(img_size: int | dict | tuple = 518):
    """
    Eval-time transform for DINO unsupervised analysis:
    - Resize with aspect ratio preserved if img_size is int/dict
    - ToTensor
    - ImageNet normalization
    """
    resize_spec = parse_img_size_config(img_size)

    return T.Compose(
        [
            _ResizeLongestEdge(resize_spec),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def openclip_native_transforms(img_size: int | dict | tuple = 518):
    """
    Eval-time transform matching OpenCLIP normalization while keeping the repo's resize policy.
    """
    resize_spec = parse_img_size_config(img_size)
    return T.Compose(
        [
            _ResizeLongestEdge(resize_spec),
            T.ToTensor(),
            T.Normalize(mean=OPENCLIP_MEAN, std=OPENCLIP_STD),
        ]
    )
