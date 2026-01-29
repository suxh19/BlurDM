# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
from torch import nn


class ResMLP(nn.Module):
    def __init__(self, n_feats: int = 512):
        super().__init__()
        self.resmlp = nn.Sequential(
            nn.Linear(n_feats, n_feats),
            nn.LeakyReLU(0.1, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resmlp(x)


class denoise(nn.Module):
    def __init__(self, n_feats: int = 64, n_denoise_res: int = 5, timesteps: int = 5):
        super().__init__()
        self.max_period = timesteps
        n_featsx4 = 4 * n_feats
        resmlp = [
            nn.Linear(n_featsx4 * 2 + 1, n_featsx4),
            nn.LeakyReLU(0.1, True),
        ]
        for _ in range(n_denoise_res):
            resmlp.append(ResMLP(n_featsx4))
        self.resmlp = nn.Sequential(*resmlp)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        t = t.float() / self.max_period
        t = t.view(-1, 1)
        c = torch.cat([c, t, x], dim=1)
        return self.resmlp(c)

