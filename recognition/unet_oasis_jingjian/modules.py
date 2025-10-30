# modules.py
"""
UNetSmall - compact UNet implementation (2D).
Model returns logits (no sigmoid); loss functions apply sigmoid where needed.
"""
import torch
import torch.nn as nn
from typing import Sequence

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        return self.pool(self.conv(x)), self.conv(x)

class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # use conv transpose to upsample
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)  # after concat channels = in_ch
    def forward(self, x, skip):
        x = self.up(x)
        # If shapes mismatch due to odd sizes, center-crop skip to x
        if x.shape[-2:] != skip.shape[-2:]:
            # simple center crop of skip
            sh, sw = skip.shape[-2:]
            th, tw = x.shape[-2:]
            top = (sh - th) // 2
            left = (sw - tw) // 2
            skip = skip[..., top:top+th, left:left+tw]
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNetSmall(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, features: Sequence[int] = (32, 64, 128)):
        super().__init__()
        # encoder
        self.downs = nn.ModuleList()
        ch = in_ch
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f
        self.pool = nn.MaxPool2d(2)
        # bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        # decoder - use reversed features
        self.ups = nn.ModuleList()
        rev = list(reversed(features))
        prev = features[-1]*2
        for f in rev:
            self.ups.append(Up(prev, f))
            prev = f
        self.final = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        skips = []
        for conv in self.downs:
            x = conv(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)
        out = self.final(x)
        return out  # logits

if __name__ == "__main__":
    # quick smoke test
    model = UNetSmall(in_ch=1, out_ch=1)
    x = torch.randn(2,1,128,128)
    y = model(x)
    print("UNetSmall output shape:", y.shape)
