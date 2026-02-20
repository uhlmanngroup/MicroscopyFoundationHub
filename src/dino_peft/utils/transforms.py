from torchvision import transforms as T
import torch 
import torch.nn.functional as F
from PIL import Image
import numpy as np

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


def _shift_chw_reflect(image: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    if dx == 0 and dy == 0:
        return image
    _, h, w = image.shape
    pad_x = abs(dx)
    pad_y = abs(dy)
    pad_mode = "reflect"
    if (h == 1 and pad_y > 0) or (w == 1 and pad_x > 0):
        pad_mode = "replicate"
    padded = F.pad(
        image.unsqueeze(0),
        (pad_x, pad_x, pad_y, pad_y),
        mode=pad_mode,
    ).squeeze(0)
    y0 = pad_y - dy
    x0 = pad_x - dx
    return padded[:, y0:y0 + h, x0:x0 + w]


def _shift_hw_zero(mask: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    if dx == 0 and dy == 0:
        return mask
    h, w = mask.shape
    pad_x = abs(dx)
    pad_y = abs(dy)
    padded = F.pad(
        mask.unsqueeze(0).unsqueeze(0),
        (pad_x, pad_x, pad_y, pad_y),
        mode="constant",
        value=0,
    ).squeeze(0).squeeze(0)
    y0 = pad_y - dy
    x0 = pad_x - dx
    return padded[y0:y0 + h, x0:x0 + w]


def em_seg_online_augment(
    images: torch.Tensor,
    masks: torch.Tensor,
    *,
    prob: float = 0.5,
    max_shift_percent: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Online EM segmentation augmentation on batched tensors.

    For each sample with probability `prob`, apply:
      1) random flip axis from {h, v, hv}
      2) random shift up to max_shift_percent per axis
    """
    if images.dim() != 4 or masks.dim() != 3:
        raise ValueError(
            f"Expected images=(B,C,H,W) and masks=(B,H,W), got {tuple(images.shape)} and {tuple(masks.shape)}"
        )
    p = float(min(max(prob, 0.0), 1.0))
    if p == 0.0:
        return images, masks, 0
    if max_shift_percent < 0 or max_shift_percent >= 1:
        raise ValueError("max_shift_percent must be in [0, 1).")

    bsz = images.size(0)
    apply_mask = torch.rand(bsz, device=images.device) < p
    selected = torch.nonzero(apply_mask, as_tuple=False).flatten()
    if selected.numel() == 0:
        return images, masks, 0

    out_images = images.clone()
    out_masks = masks.clone()
    _, _, h, w = images.shape
    max_dx = int(round(w * float(max_shift_percent)))
    max_dy = int(round(h * float(max_shift_percent)))

    for i in selected.tolist():
        axis_mode = int(torch.randint(0, 3, (1,), device=images.device).item())
        if axis_mode in (0, 2):
            out_images[i] = torch.flip(out_images[i], dims=(-1,))
            out_masks[i] = torch.flip(out_masks[i], dims=(-1,))
        if axis_mode in (1, 2):
            out_images[i] = torch.flip(out_images[i], dims=(-2,))
            out_masks[i] = torch.flip(out_masks[i], dims=(-2,))

        dx = int(torch.randint(-max_dx, max_dx + 1, (1,), device=images.device).item()) if max_dx > 0 else 0
        dy = int(torch.randint(-max_dy, max_dy + 1, (1,), device=images.device).item()) if max_dy > 0 else 0
        if dx != 0 or dy != 0:
            out_images[i] = _shift_chw_reflect(out_images[i], dx=dx, dy=dy)
            out_masks[i] = _shift_hw_zero(out_masks[i], dx=dx, dy=dy)

    return out_images, out_masks, int(selected.numel())

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
