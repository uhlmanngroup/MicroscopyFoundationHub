from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from dino_peft.config import load_paths

from .base import BackboneAdapter, BackboneOutput

_WEIGHT_TAGS = {
    "imagenet": ResNet50_Weights.IMAGENET1K_V1,
    "imagenet1k": ResNet50_Weights.IMAGENET1K_V1,
    "imagenet1k_v1": ResNet50_Weights.IMAGENET1K_V1,
    "imagenet1k_v2": ResNet50_Weights.IMAGENET1K_V2,
}


def _checkpoint_dir() -> Path | None:
    """``${models_root}/resnet50``, next to the DINOv3 checkpoints.

    Left to itself, torchvision downloads into ``$TORCH_HOME`` (``~/.cache/torch``),
    which on the cluster is the small NFS home rather than the scratch directory
    every other backbone's weights live in.
    """
    models_root = load_paths().get("models_root")
    if not models_root:
        return None
    return Path(models_root).expanduser() / "resnet50"


def _load_state_dict(w: ResNet50_Weights | None, weights: str | None) -> dict:
    """ImageNet weights from an explicit path, else downloaded under ``models_root``."""
    if weights:
        weights_path = Path(weights).expanduser()
        if not weights_path.exists():
            raise FileNotFoundError(f"ResNet50 weights not found: {weights_path}")
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict):
            state = state.get("state_dict", state.get("model", state))
        print(f"[resnet50] weights <- {weights_path}")
        return state

    model_dir = _checkpoint_dir()
    if model_dir is None:
        print("[resnet50] models_root undefined; falling back to the torchvision cache")
        return w.get_state_dict(progress=False)

    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        state = torch.hub.load_state_dict_from_url(
            w.url, model_dir=str(model_dir), map_location="cpu", progress=False
        )
    except OSError as exc:
        # models_root is a cluster path; off the cluster it may not be creatable.
        print(f"[resnet50] {model_dir} unusable ({exc}); falling back to the torchvision cache")
        return w.get_state_dict(progress=False)
    print(f"[resnet50] weights <- {model_dir / Path(w.url).name}")
    return state


class ResNet50Adapter(BackboneAdapter):
    def __init__(self,
                 variant: str,
                 device: str | torch.device,
                 *,
                 pretrained: str | None = "imagenet",
                 weights: str | None = None,
                   ) -> None:
        super().__init__(name="resnet50", variant=variant)

        tag = (pretrained or "").lower()
        if tag in ("", "none", "null", "random"):
            w = None
        elif tag in _WEIGHT_TAGS:
            w = _WEIGHT_TAGS[tag]
        else:
            raise ValueError(f"Unknown pretrained tag '{pretrained}' for resnet50.")

        # Built without weights, then loaded separately, so that the checkpoint is
        # fetched into models_root instead of torchvision's own cache. The random
        # ablation (pretrained: random) loads nothing at all.
        self.model = resnet50(weights=None)
        if w is not None or weights:
            self.model.load_state_dict(_load_state_dict(w, weights))

        self.model.avgpool = nn.Identity()
        self.model.fc = nn.Identity()
        self.embed_dim = 2048
        self.patch_size = 32
        self.to(device)
        self.eval()

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        m = self.model
        x = m.relu(m.bn1(m.conv1(x)))
        s2 = x                    # (B,   64, H/2,  W/2)  
        x = m.maxpool(x)
        s4 = m.layer1(x)          # (B,  256, H/4,  W/4)
        s8 = m.layer2(s4)         # (B,  512, H/8,  W/8)
        s16 = m.layer3(s8)        # (B, 1024, H/16, W/16)
        s32 = m.layer4(s16)       # (B, 2048, H/32, W/32)
        
        gh, gw = s32.shape[-2], s32.shape[-1]
        patch_tokens = s32.flatten(2).transpose(1,2) # (B, gh*gw, 2048)
        return BackboneOutput(
            global_embedding=patch_tokens.mean(dim=1),
            patch_tokens=patch_tokens,
            grid_size=(gh, gw),
            feature_maps=[s2, s4, s8, s16, s32]
        )
