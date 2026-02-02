"""Angle (limited-angle CT) encoder for Stage2.

Stage1 uses LE_arch(x_deg, x_gt) to produce a *teacher* prior Z^S.
Stage2 needs an encoder that takes *only* the degraded observation x_deg and
produces a condition latent Z^B (analogous to BlurDM's Blur Encoder).

This file implements a lightweight encoder that mirrors LE_arch's backbone but
removes the GT concatenation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def default_conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class ResBlock(nn.Module):
    def __init__(
        self,
        conv,
        n_feats: int,
        kernel_size: int,
        bias: bool = True,
        bn: bool = False,
        act: nn.Module | None = None,
    ):
        super().__init__()
        if act is None:
            act = nn.LeakyReLU(0.1, inplace=True)

        layers = []
        for i in range(2):
            layers.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                layers.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                layers.append(act)
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class AngleEncoder(nn.Module):
    """Encoder that maps a degraded CT image to a 1x1 latent vector (B, C).

    Output dimension is (n_feats * 4), matching Stage1's LE_arch default and
    MIMO-UNet's prior projection (Linear(256, 2*channel)).
    """

    def __init__(
        self,
        n_feats: int = 64,
        n_encoder_res: int = 6,
        bn: bool = False,
        in_channels: int = 1,
        pixel_unshuffle_factor: int = 4,
    ):
        super().__init__()
        unshuffle_ch = in_channels * (pixel_unshuffle_factor**2)
        conv_in_ch = unshuffle_ch

        E1 = [
            nn.Conv2d(conv_in_ch, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
        ]
        E2 = [ResBlock(default_conv, n_feats, kernel_size=3, bn=bn) for _ in range(n_encoder_res)]
        E3 = [
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        ]

        self.E = nn.Sequential(*(E1 + E2 + E3))
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True),
        )
        self.pixel_unshuffle = nn.PixelUnshuffle(pixel_unshuffle_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,1,H,W) in [-0.5, 0.5] -> z: (B, C)."""
        x0 = self.pixel_unshuffle(x)
        fea = self.E(x0).squeeze(-1).squeeze(-1)
        return self.mlp(fea)
