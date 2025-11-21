from torchvision import transforms as T
import torch 
from PIL import Image

from .image_size import compute_resized_hw, parse_img_size_config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def em_seg_transforms(img_size=(308,308)):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

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

def em_dino_unsup_transforms(img_size: int | dict | tuple = 518):
    """
    Eval-time transform for DINO unsupervised analysis:
    - Resize with aspect ratio preserved if img_size is int/dict
    - ToTensor
    - ImageNet normalization
    """
    resize_spec = parse_img_size_config(img_size)

    def _resize(image: Image.Image) -> Image.Image:
        target_hw = compute_resized_hw((image.height, image.width), resize_spec)
        target_wh = (target_hw[1], target_hw[0])
        if image.size != target_wh:
            return image.resize(target_wh, Image.BICUBIC)
        return image

    return T.Compose(
        [
            T.Lambda(_resize),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
