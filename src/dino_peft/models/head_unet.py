import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x
    
class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels=(64, 256, 512, 1024, 2048),
                 num_classes: int = 2, 
                 decoder_channels=(256, 128, 64, 32, 16)):
        super().__init__()
        enc = list(encoder_channels)
        in_chs = [enc[-1]] + list(decoder_channels[:-1])
        skip_chs = list(reversed(enc[:-1])) + [0]
        self.blocks = nn.ModuleList([
            DecoderBlock(i, s, o)
            for i, s, o in zip(in_chs, skip_chs, decoder_channels)
        ])
        self.head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, features, out_hw):
        skips = list(reversed(features[:-1]))
        x = features[-1]
        for i, block in enumerate(self.blocks):
            x = block(x, skips[i] if i < len(skips) else None)
        logits = self.head(x)
        return F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)