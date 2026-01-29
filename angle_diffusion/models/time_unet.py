from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    groups = min(max_groups, num_channels)
    while groups > 1 and (num_channels % groups) != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal timestep embedding, returns [B, dim].
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000.0) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_dim: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = _group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.temb_proj = nn.Linear(temb_dim, out_ch)
        self.norm2 = _group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb_proj(F.silu(temb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class TimeConditionedUNet(nn.Module):
    """
    Minimal time-conditioned UNet for CT (single-channel by default).

    The model predicts x0 from x_t given timestep t (severity).
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        max_t: int = 100,
    ) -> None:
        super().__init__()
        self.max_t = max_t

        temb_dim = base_channels * 4
        self._temb_dim_in = base_channels  # Store for forward pass
        self.temb = nn.Sequential(
            nn.Linear(base_channels, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )

        chs = [base_channels * m for m in channel_mults]
        self.conv_in = nn.Conv2d(in_channels, chs[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = chs[0]
        for level, out_ch in enumerate(chs):
            blocks = nn.ModuleList([ResBlock(in_ch, out_ch, temb_dim)])
            for _ in range(num_res_blocks - 1):
                blocks.append(ResBlock(out_ch, out_ch, temb_dim))
            self.down_blocks.append(blocks)
            in_ch = out_ch
            if level != len(chs) - 1:
                self.downsamples.append(Downsample(in_ch))

        self.mid1 = ResBlock(in_ch, in_ch, temb_dim)
        self.mid2 = ResBlock(in_ch, in_ch, temb_dim)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level, out_ch in enumerate(reversed(chs)):
            skip_ch = out_ch
            blocks = nn.ModuleList([ResBlock(in_ch + skip_ch, out_ch, temb_dim)])
            for _ in range(num_res_blocks - 1):
                blocks.append(ResBlock(out_ch, out_ch, temb_dim))
            self.up_blocks.append(blocks)
            in_ch = out_ch
            if level != len(chs) - 1:
                self.upsamples.append(Upsample(in_ch))

        self.norm_out = _group_norm(in_ch)
        self.conv_out = nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Normalize t into a stable range for embedding.
        t = t.clamp(min=0, max=self.max_t).to(dtype=torch.float32) / float(self.max_t)
        temb = timestep_embedding(t, self._temb_dim_in)
        temb = self.temb(temb)

        h = self.conv_in(x)
        skips: list[torch.Tensor] = []
        for level, blocks in enumerate(self.down_blocks):
            for blk in blocks:  # type: ignore[union-attr]
                h = blk(h, temb)
            skips.append(h)
            if level != len(self.down_blocks) - 1:
                h = self.downsamples[level](h)

        h = self.mid1(h, temb)
        h = self.mid2(h, temb)

        for level, blocks in enumerate(self.up_blocks):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            for blk in blocks:  # type: ignore[union-attr]
                h = blk(h, temb)
            if level != len(self.up_blocks) - 1:
                h = self.upsamples[level](h)

        h = self.conv_out(F.silu(self.norm_out(h)))
        return h
