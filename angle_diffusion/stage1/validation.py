"""验证逻辑"""

from __future__ import annotations

import os
from typing import TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from angle_diffusion.config import Stage1Config
from angle_diffusion.utils.misc import ensure_dir

from .utils import shift_to_unit
from .visualization import save_visualizations


class ValidationMetrics(TypedDict):
    """验证指标类型定义"""
    val_l1: float
    val_psnr: float
    val_ssim: float | None


@torch.no_grad()
def validate(
    cfg: Stage1Config,
    model: torch.nn.Module,
    model_le: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int | None = None,
    out_dir: str | None = None,
) -> ValidationMetrics:
    """使用预计算的 SART 图像进行验证
    
    Args:
        cfg: Stage1 配置
        model: MIMO-UNet 解码器模型
        model_le: LatentEncoder 模型
        loader: 验证数据加载器
        device: 计算设备
        epoch: 当前轮次（可选）
        out_dir: 输出目录（可选）
        
    Returns:
        包含验证指标的字典（val_l1, val_psnr, val_ssim）
    """
    model.eval()
    model_le.eval()
    losses = []
    psnr_vals = []
    ssim_vals = []

    try:
        from skimage.metrics import structural_similarity as skimage_ssim
    except Exception:
        skimage_ssim = None
        print(
            "[Stage1] Warning: skimage not available, SSIM will be omitted for this validation."
        )

    # 可视化存储
    vis_images = []

    # 根据 val_ratio 计算需要使用的数据量
    total_batches = len(loader)
    num_batches = max(1, int(total_batches * cfg.val_ratio))
    max_ssim_samples = max(0, int(cfg.val_vis_samples))
    ssim_count = 0

    for batch_idx, (x0, x_t) in enumerate(loader):
        if batch_idx >= num_batches:
            break

        x0 = x0.to(device)  # [B,1,H,W] GT in [0,1]
        x_t = x_t.to(device)  # [B,1,H,W] SART input
        bs = x0.shape[0]

        # Extract latent prior
        z_pred = model_le(x_t, x0)
        # Get multi-scale predictions
        outputs = model(x_t, z_pred)
        # Use full-scale output for validation
        pred = outputs[2]

        x0_metric = shift_to_unit(x0)
        pred_metric = shift_to_unit(pred)
        l1_per = F.l1_loss(pred_metric, x0_metric, reduction="none").mean(
            dim=(1, 2, 3)
        )
        mse_per = F.mse_loss(pred_metric, x0_metric, reduction="none").mean(
            dim=(1, 2, 3)
        )
        psnr_per = 10.0 * torch.log10(1.0 / (mse_per + 1e-12))
        losses.extend(l1_per.detach().cpu().tolist())
        psnr_vals.extend(psnr_per.detach().cpu().tolist())

        if skimage_ssim is not None and ssim_count < max_ssim_samples:
            for i in range(min(bs, max_ssim_samples - ssim_count)):
                gt_np = x0_metric[i].squeeze().detach().cpu().numpy()
                pred_np = pred_metric[i].squeeze().detach().cpu().numpy()
                ssim_result = skimage_ssim(gt_np, pred_np, data_range=1.0)
                # skimage_ssim 可能返回元组 (ssim, diff)，我们只需要 ssim 值
                if isinstance(ssim_result, tuple):
                    ssim_val = float(ssim_result[0])
                else:
                    ssim_val = float(ssim_result)
                ssim_vals.append(ssim_val)
                ssim_count += 1

        # 收集可视化样本
        if cfg.val_visualize and len(vis_images) < cfg.val_vis_samples:
            # 每个样本保存：原图、退化图、预测图
            for i in range(min(bs, cfg.val_vis_samples - len(vis_images))):
                vis_images.append(
                    {
                        "gt": x0[i].cpu(),
                        "degraded": x_t[i].cpu(),
                        "pred": pred[i].cpu(),
                    }
                )

    # 保存可视化结果
    if (
        cfg.val_visualize
        and vis_images
        and epoch is not None
        and out_dir is not None
    ):
        save_visualizations(out_dir, vis_images, epoch)

    metrics: ValidationMetrics = {
        "val_l1": float(np.mean(losses)) if losses else 0.0,
        "val_psnr": float(np.mean(psnr_vals)) if psnr_vals else 0.0,
        "val_ssim": float(np.mean(ssim_vals)) if ssim_vals else None,
    }

    if epoch is not None and out_dir is not None:
        logs_dir = ensure_dir(os.path.join(out_dir, "logs"))
        metrics_path = os.path.join(logs_dir, "metrics.csv")
        is_new = not os.path.exists(metrics_path)
        with open(metrics_path, "a", encoding="utf-8") as f:
            if is_new:
                f.write("epoch,val_l1,val_psnr,val_ssim\n")
            val_ssim_str = (
                "" if metrics["val_ssim"] is None else f"{metrics['val_ssim']:.6f}"
            )
            f.write(
                f"{epoch},{metrics['val_l1']:.6f},{metrics['val_psnr']:.6f},{val_ssim_str}\n"
            )

    return metrics
