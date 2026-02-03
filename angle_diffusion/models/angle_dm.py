"""Angle Diffusion Model (Stage2) in latent space.

We follow BlurDM's Stage2 idea (latent prior generation), but for limited-angle CT.

Case A assumption:
    The CT degradation operator already outputs reconstructions on a consistent
    intensity scale across degradation levels. Therefore we do *not* need the
    exposure-style alpha normalization used in BlurDM. In the reverse update,
    we can set alpha_t=1 for all t.

This module learns two estimators (both MLPs) operating on latent vectors:
    - a_theta: artifact/angle-residual estimator
    - eps_theta: Gaussian noise estimator

Reverse update (alpha=1):
    z_{t-1} = z_t - a_hat(z_t,t,zB) - (\bar\beta_t-\bar\beta_{t-1}) * eps_hat(z_t,t,zB)

One-step forward (as in BlurDM Eq.(7) in latent space):
    z_T = z_B + \bar\beta_T * eps

Reference: BlurDM paper, Eq.(12) with alpha_t/alpha_{t-1}=1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _mlp(in_dim: int, out_dim: int, hidden_dim: int, depth: int = 6) -> nn.Sequential:
    """Simple LeakyReLU MLP used by residual estimators."""
    layers = []
    d = in_dim
    for i in range(depth - 1):
        layers.append(nn.Linear(d, hidden_dim))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


@dataclass
class AngleDMSchedule:
    T: int = 5
    beta_max: float = 0.02


class AngleDM(nn.Module):
    # Type hints for registered buffers
    beta_bar: torch.Tensor

    def __init__(
        self,
        latent_dim: int = 256,
        T: int = 5,
        beta_max: float = 0.02,
        time_embed_dim: int = 128,
        hidden_dim: int = 512,
        use_degrade_level_cond: bool = True,
        level_cond_dim: int = 1,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.T = int(T)
        self.beta_max = float(beta_max)
        self.time_embed_dim = int(time_embed_dim)
        self.use_degrade_level_cond = bool(use_degrade_level_cond)
        self.level_cond_dim = int(level_cond_dim)

        # \bar\beta schedule: [0..T], linear is sufficient for our latent prior training.
        beta_bar = torch.linspace(0.0, self.beta_max, self.T + 1)
        self.register_buffer("beta_bar", beta_bar, persistent=False)

        # Time embedding for diffusion step index (1..T)
        self.t_embed = nn.Embedding(self.T + 1, self.time_embed_dim)

        # Optional embedding for CT degradation condition.
        # Default: scalar t in [0,100].
        # Extended: vector [t, angle_feat...] where the first dim is t in [0,100]
        # and the remaining dims are normalized to [0,1].
        if self.use_degrade_level_cond:
            self.level_mlp = nn.Sequential(
                nn.Linear(self.level_cond_dim, self.time_embed_dim),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(self.time_embed_dim, self.time_embed_dim),
                nn.LeakyReLU(0.1, inplace=True),
            )
        else:
            self.level_mlp = None

        # Two estimators: artifact residual and noise residual.
        # Input: [z_t, z_cond, t_emb, (optional level_emb)]
        cond_dim = self.latent_dim * 2 + self.time_embed_dim
        if self.use_degrade_level_cond:
            cond_dim += self.time_embed_dim

        self.artifact_est = _mlp(
            cond_dim, self.latent_dim, hidden_dim=hidden_dim, depth=6
        )
        self.noise_est = _mlp(cond_dim, self.latent_dim, hidden_dim=hidden_dim, depth=6)

    def _build_cond(
        self,
        z_t: torch.Tensor,
        step: int,
        z_cond: torch.Tensor,
        degrade_level: torch.Tensor | None,
    ) -> torch.Tensor:
        B = z_t.shape[0]
        step_ids = torch.full((B,), int(step), device=z_t.device, dtype=torch.long)
        t_emb = self.t_embed(step_ids)

        parts = [z_t, z_cond, t_emb]
        if self.use_degrade_level_cond:
            if degrade_level is None:
                # If not provided, assume worst-case (t=100) and zero angle feature.
                base = torch.full((B, 1), 100.0, device=z_t.device)
                if self.level_cond_dim > 1:
                    pad = torch.zeros(
                        (B, self.level_cond_dim - 1),
                        device=z_t.device,
                        dtype=base.dtype,
                    )
                    degrade_level = torch.cat([base, pad], dim=1)
                else:
                    degrade_level = base

            if degrade_level.dim() == 1:
                degrade_level = degrade_level.view(B, 1)
            elif degrade_level.dim() != 2:
                raise ValueError(
                    f"degrade_level 期望 1D/2D Tensor，实际维度: {degrade_level.dim()}"
                )

            if degrade_level.shape[0] != B:
                raise ValueError(
                    f"degrade_level batch 维度不匹配: got {degrade_level.shape[0]}, expected {B}"
                )

            if degrade_level.dtype != torch.float32:
                degrade_level = degrade_level.float()

            if degrade_level.shape[1] != self.level_cond_dim:
                if degrade_level.shape[1] == 1 and self.level_cond_dim > 1:
                    # Backward-compatible: scalar t -> pad zeros for angle_feat.
                    pad = torch.zeros(
                        (B, self.level_cond_dim - 1),
                        device=degrade_level.device,
                        dtype=degrade_level.dtype,
                    )
                    degrade_level = torch.cat([degrade_level, pad], dim=1)
                else:
                    raise ValueError(
                        f"degrade_level cond_dim 不匹配: got {degrade_level.shape[1]}, expected {self.level_cond_dim}"
                    )

            # Normalize to [0,1] for stability:
            # - dim0: t in [0,100] -> /100
            # - dim1..: already in [0,1]
            t_part = (degrade_level[:, 0:1] / 100.0).clamp(0.0, 1.0)
            if self.level_cond_dim > 1:
                extra = degrade_level[:, 1:].clamp(0.0, 1.0)
                lv = torch.cat([t_part, extra], dim=1)
            else:
                lv = t_part
            if self.level_mlp is not None:
                lv_emb = self.level_mlp(lv)
                parts.append(lv_emb)

        return torch.cat(parts, dim=1)

    def predict(
        self,
        z_t: torch.Tensor,
        step: int,
        z_cond: torch.Tensor,
        degrade_level: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict artifact residual and noise residual at a given reverse step."""
        cond = self._build_cond(z_t, step, z_cond, degrade_level)
        a_hat = self.artifact_est(cond)
        eps_hat = self.noise_est(cond)
        return a_hat, eps_hat

    def reverse_from(
        self,
        z_T: torch.Tensor,
        z_cond: torch.Tensor,
        degrade_level: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Deterministic reverse process from z_T to z_0 (Case A: alpha=1)."""
        z = z_T
        for step in range(self.T, 0, -1):
            a_hat, eps_hat = self.predict(z, step, z_cond, degrade_level=degrade_level)
            delta = (self.beta_bar[step] - self.beta_bar[step - 1]).view(1)
            z = z - a_hat - delta * eps_hat
        return z

    def sample(
        self,
        z_cond: torch.Tensor,
        degrade_level: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convenience sampling: start from z_T = z_cond + beta_bar[T]*eps."""
        if noise is None:
            noise = torch.randn_like(z_cond)
        z_T = z_cond + self.beta_bar[self.T] * noise
        return self.reverse_from(z_T, z_cond, degrade_level=degrade_level)
