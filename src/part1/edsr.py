"""Compact EDSR baseline, 16 residual blocks and 64 features.

This matches the official EDSR-PyTorch baseline (Lim et al.) and the
DIV2K 4× checkpoint edsr_baseline_x4. Inference only; the mandatory path
does not train it.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn


def _conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)


class _MeanShift(nn.Conv2d):
    def __init__(self, rgb_range: float, sign: int = -1) -> None:
        super().__init__(3, 3, kernel_size=1)
        rgb_mean = (0.4488, 0.4371, 0.4040)
        std = torch.ones(3)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.tensor(rgb_mean) / std
        for parameter in self.parameters():
            parameter.requires_grad = False


class _ResBlock(nn.Module):
    def __init__(self, n_feats: int, res_scale: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            _conv(n_feats, n_feats, 3),
            nn.ReLU(True),
            _conv(n_feats, n_feats, 3),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class _Upsampler(nn.Sequential):
    def __init__(self, scale: int, n_feats: int) -> None:
        layers: list[nn.Module] = []
        if scale & (scale - 1):
            raise ValueError("scale must be a power of two")
        for _ in range(int(math.log(scale, 2))):
            layers.append(_conv(n_feats, 4 * n_feats, 3))
            layers.append(nn.PixelShuffle(2))
        super().__init__(*layers)


class EDSRBaseline(nn.Module):
    def __init__(self, scale: int = 4, n_resblocks: int = 16, n_feats: int = 64, rgb_range: float = 255) -> None:
        super().__init__()
        self.sub_mean = _MeanShift(rgb_range)
        self.add_mean = _MeanShift(rgb_range, sign=1)
        self.head = nn.Sequential(_conv(3, n_feats, 3))
        self.body = nn.Sequential(
            *[_ResBlock(n_feats, res_scale=1) for _ in range(n_resblocks)],
            _conv(n_feats, n_feats, 3),
        )
        self.tail = nn.Sequential(_Upsampler(scale, n_feats), _conv(n_feats, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sub_mean(x)
        x = self.head(x)
        residual = self.body(x)
        residual = residual + x
        x = self.tail(residual)
        return self.add_mean(x)


def load_edsr(checkpoint: Path, device: torch.device) -> EDSRBaseline:
    model = EDSRBaseline()
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model
