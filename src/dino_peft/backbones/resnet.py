from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50
from .base import BackboneAdapter, BackboneOutput


class ResNet50Adapter(BackboneAdapter):
    def __init__(self, 
                 variant: str, 
                 device: str | torch.device, 
                 *,
                 pretrained: str | None = "imagenet", 
                   ) -> None:
        super().__init__(name="resnet50", variant=variant)

        tag = (pretrained or "").lower()
        if tag in ("", "none", "null", "random"):
            w = None
        elif tag in ("imagenet", "imagenet1k", "imagenet1k_v1"):
            w = ResNet50_Weights.IMAGENET1K_V1
        elif tag == "imagenet1k_v2":
            w = ResNet50_Weights.IMAGENET1K_V2
        else:
            raise ValueError(f"Unknown pretrained tag '{pretrained}' for resnet50.")
        
        self.model = resnet50(weights=w)
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
