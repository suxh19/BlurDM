# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Latent Angle Degradation Diffusion (AngleDM)

This implements a BlurDM-style dual diffusion in latent space:
  - a learned "angle residual" (artifact) term
  - a learned Gaussian noise term

Angle schedule:
  We map angle range (phi) -> a monotonic "scale" (alpha) so we can reuse the
  same closed-form accumulation used in LatentExposureDiffusion.

  Given phi_0 (full) > ... > phi_T (limited), we define:
      alpha_t = phi_0 / phi_t   (alpha increases as angles decrease)
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .LatentBlurDM import denoise
from .LatentEncoder import LE_arch


class LatentAngleDiffusion(nn.Module):
    def __init__(
        self,
        total_timestamps: int = 5,
        *,
        focus_table_path: Optional[str] = None,
        phi_max: float = 180.0,
        phi_min: float = 60.0,
        phi_schedule: Optional[Sequence[float]] = None,
        beta_start: float = 0.0,
        beta_end: float = 0.02,
        in_channels: int = 3,
        pixel_unshuffle_factor: int = 4,
        spvised_mid_out: bool = False,
    ) -> None:
        super().__init__()

        self.angle_model = denoise(timesteps=total_timestamps)
        self.noise_model = denoise(timesteps=total_timestamps)
        self.condition_encoder = LE_arch(
            in_channels=in_channels,
            pixel_unshuffle_factor=pixel_unshuffle_factor,
        )

        self.total_timestamps = total_timestamps
        self.spvised_mid_out = spvised_mid_out

        betas = torch.linspace(beta_start, beta_end, self.total_timestamps, dtype=torch.float32)

        if phi_schedule is None and focus_table_path is not None:
            # Focus table stores ascending ns (119..218). For diffusion we need
            # "best -> worst" (218..119), so reverse it then resample to T+1.
            data = np.load(focus_table_path)
            if "ns" not in data:
                raise KeyError(f"focus table {focus_table_path} missing key 'ns'")
            ns_desc = data["ns"][::-1].astype(np.float32)

            idx = np.linspace(0, len(ns_desc) - 1, self.total_timestamps + 1)
            idx = np.rint(idx).astype(np.int64)
            phis = torch.from_numpy(ns_desc[idx]).to(dtype=torch.float32)
        elif phi_schedule is None:
            # Full-view -> limited-angle (monotonic decreasing).
            phis = torch.linspace(phi_max, phi_min, self.total_timestamps + 1, dtype=torch.float32)
        else:
            phis = torch.tensor(list(phi_schedule), dtype=torch.float32)
            if phis.numel() != self.total_timestamps + 1:
                raise ValueError(
                    f"phi_schedule length must be total_timestamps+1="
                    f"{self.total_timestamps + 1}, got {phis.numel()}"
                )

        # Map angles -> monotonic scale (alpha). alpha_0 = 1, alpha increases with degradation.
        alphas = phis[0] / phis.clamp_min(1e-8)

        betas_bar_list = self.get_beta_bar(alphas, betas)
        time_stamps_list = torch.arange(self.total_timestamps, 0, -1, dtype=torch.long)

        self.register_buffer("phis", phis)
        self.register_buffer("alphas", alphas)
        self.register_buffer("betas_bar", betas_bar_list)
        self.register_buffer("time_stamps_list", time_stamps_list)

    def get_beta_bar(self, alphas: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
        # Same accumulation as BlurDM latent implementation.
        betas_bar_list = []
        for t in range(1, self.total_timestamps + 1):
            sub_betas = betas[:t]
            weights = torch.tensor(
                [(alphas[i - 1] / alphas[t]) ** 2 for i in range(1, t + 1)],
                dtype=torch.float32,
            )
            result = torch.sum(weights * sub_betas)
            betas_bar_list.append(result.clone().detach().sqrt())
        return torch.tensor(betas_bar_list)

    def q_sample_d(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(latent)
        return latent + self.betas_bar[self.total_timestamps - 1] * noise, noise

    def forward(self, cond: torch.Tensor):
        pred_angle_list = []
        pred_noise_list = []

        device = self.alphas.device
        b = cond.shape[0]

        # Encode condition into latent prior space (same trick as BlurDM: pass cond twice).
        T_z = self.condition_encoder(cond, cond)
        noise_latent, noise = self.q_sample_d(T_z)

        for i in self.time_stamps_list:
            t = torch.full((b,), i, device=device, dtype=torch.long)

            pred_noise = self.noise_model(noise_latent, t, T_z)
            pred_angle = self.angle_model(noise_latent, t, T_z)

            if self.spvised_mid_out:
                pred_angle_list.append(pred_angle)
                pred_noise_list.append(pred_noise)

            if int(i.item()) == 1:
                noise_cof = self.betas_bar[i - 1]
            else:
                beta_t_bar = self.betas_bar[i - 1]
                beta_t_minus1_bar = self.betas_bar[i - 2]
                noise_cof = (self.alphas[i] * beta_t_bar) / self.alphas[i - 1] - beta_t_minus1_bar

            # Reverse update: de-artifact (angle) + denoise in latent space.
            noise_latent = ((self.alphas[i] * (noise_latent) - pred_angle) / self.alphas[i - 1]) - (
                noise_cof * pred_noise
            )

        if self.spvised_mid_out:
            return noise_latent, pred_angle_list, pred_noise_list, noise
        return noise_latent
