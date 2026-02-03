"""验证逻辑"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from angle_diffusion.config import Stage2Config
from angle_diffusion.models.angle_encoder import AngleEncoder
from angle_diffusion.models.angle_dm import AngleDM
from .physics.ct_physics_batch import CT_PhysicsBatch
from angle_diffusion.utils.misc import ensure_dir

from .degradation import degrade_batch_with_cache
from .visualization import save_visualizations


@torch.no_grad()
def validate(
    cfg: Stage2Config,
    teacher: torch.nn.Module,
    be: AngleEncoder,
    dm: AngleDM,
    decoder: torch.nn.Module | None,
    loader: DataLoader,
    ct_physics: CT_PhysicsBatch,
    cache_dir: str | None,
    device: torch.device,
    epoch: int | None = None,
    out_dir: str | None = None,
) -> dict[str, float]:
    """验证模型性能

    通过计算潜在空间 L1 损失来评估模型: ||Z0^B - Z^S||_1

    Args:
        cfg: Stage2 配置
        teacher: Stage1 的教师模型（latent encoder）
        be: AngleEncoder 模型
        dm: AngleDM 模型
        decoder: Stage1 的解码器（可选，用于可视化）
        loader: 验证数据加载器
        ct_physics: CT 物理操作器
        cache_dir: 缓存目录
        device: 计算设备
        epoch: 当前轮次（可选）
        out_dir: 输出目录（可选）

    Returns:
        包含验证指标的字典
    """
    teacher.eval()
    be.eval()
    dm.eval()
    if decoder is not None:
        decoder.eval()

    losses = []
    vis_images: list[dict] = []
    max_vis = max(0, int(cfg.val_vis_samples))

    total_batches = len(loader)
    num_batches = max(1, int(total_batches * cfg.val_ratio))

    t_level = int(cfg.degrade_t_val)
    angle_feat_val = torch.from_numpy(ct_physics.get_angle_feature(t_level)).to(device)

    for batch_idx, (x0, gt_paths) in enumerate(loader):
        if batch_idx >= num_batches:
            break

        if isinstance(gt_paths, (list, tuple)):
            paths = [str(p) for p in gt_paths]
        else:
            paths = [str(gt_paths)]

        gt_01 = (x0 + 0.5).clamp(0.0, 1.0)
        gt_01_np = gt_01.numpy().astype(np.float32, copy=False)
        deg_np = degrade_batch_with_cache(
            ct_physics,
            gt_01_np,
            paths,
            t_level,
            cache_dir,
            clip_01=True,
        )
        deg_shift = deg_np - 0.5
        x0 = x0.to(device)
        x_deg = torch.from_numpy(deg_shift).to(device)
        B = int(x0.shape[0])
        t_tensor = torch.full((B, 1), float(t_level), device=device)
        angle_feat = angle_feat_val.unsqueeze(0).expand(B, -1)
        degrade_cond = torch.cat([t_tensor, angle_feat], dim=1)

        # Teacher sharp prior (from Stage1 SE): Z^S = LE_arch(x_deg, x0)
        z_s = teacher(x_deg, x0)

        # Condition latent from degraded only: Z^B
        z_b = be(x_deg)

        # One-step forward: Z_T = Z_B + beta_bar[T]*eps
        z0_pred = dm.sample(z_b, degrade_level=degrade_cond)

        loss = F.l1_loss(z0_pred, z_s)
        losses.append(float(loss.detach().cpu().item()))

        if (
            cfg.val_visualize
            and decoder is not None
            and epoch is not None
            and out_dir is not None
            and len(vis_images) < max_vis
        ):
            outputs = decoder(x_deg, z0_pred)
            if isinstance(outputs, (list, tuple)):
                pred = outputs[-1]
            else:
                pred = outputs
            pred = pred.clamp(-0.5, 0.5)

            bs = x0.shape[0]
            for i in range(min(bs, max_vis - len(vis_images))):
                vis_images.append(
                    {
                        "gt": x0[i].detach().cpu(),
                        "degraded": x_deg[i].detach().cpu(),
                        "pred": pred[i].detach().cpu(),
                    }
                )

    val_l1 = float(np.mean(losses)) if losses else 0.0

    if epoch is not None and out_dir is not None:
        logs_dir = ensure_dir(os.path.join(out_dir, "logs"))
        metrics_path = os.path.join(logs_dir, "metrics.csv")
        is_new = not os.path.exists(metrics_path)
        with open(metrics_path, "a", encoding="utf-8") as f:
            if is_new:
                f.write("epoch,val_l1\n")
            f.write(f"{epoch},{val_l1:.6f}\n")

        if cfg.val_visualize and vis_images:
            save_visualizations(out_dir, vis_images, epoch)

    return {"val_l1": val_l1}
