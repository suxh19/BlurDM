"""验证过程中的可视化功能"""

from __future__ import annotations

import os

import numpy as np

from angle_diffusion.utils.misc import ensure_dir
from .utils import shift_to_unit_np


def save_visualizations(
    out_dir: str,
    vis_images: list[dict],
    epoch: int,
) -> None:
    """保存验证可视化结果
    
    Args:
        out_dir: 输出目录
        vis_images: 包含 gt, degraded, pred 的字典列表
        epoch: 当前轮次
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    try:
        from skimage.metrics import structural_similarity as skimage_ssim
    except Exception:
        skimage_ssim = None

    vis_dir = os.path.join(out_dir, "logs", "visualizations", f"epoch_{epoch:04d}")
    ensure_dir(vis_dir)

    for idx, img_dict in enumerate(vis_images):
        gt = img_dict["gt"].squeeze().numpy()  # [H, W]
        degraded = img_dict["degraded"].squeeze().numpy()
        pred = img_dict["pred"].squeeze().numpy()

        # 创建对比图
        fig = plt.figure(figsize=(15, 5))
        gs = GridSpec(1, 3, figure=fig, wspace=0.3)

        # Ground Truth
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(gt, cmap="gray", vmin=0, vmax=1)
        ax1.set_title("Ground Truth", fontsize=12, fontweight="bold")
        ax1.axis("off")
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        # SART Input
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(degraded, cmap="gray", vmin=0, vmax=1)
        ax2.set_title("SART Input", fontsize=12, fontweight="bold")
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        # Prediction
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(pred, cmap="gray", vmin=0, vmax=1)
        ax3.set_title("Prediction", fontsize=12, fontweight="bold")
        ax3.axis("off")
        plt.colorbar(im3, ax=ax3, fraction=0.046)

        # 计算误差指标
        gt_metric = shift_to_unit_np(gt)
        pred_metric = shift_to_unit_np(pred)
        mae = np.mean(np.abs(pred_metric - gt_metric))
        mse = np.mean((pred_metric - gt_metric) ** 2)
        psnr = 10.0 * np.log10(1.0 / (mse + 1e-12))

        if skimage_ssim is not None:
            ssim_val = skimage_ssim(gt_metric, pred_metric, data_range=1.0)
            title = (
                f"Sample {idx+1} | MAE: {mae:.4f} | MSE: {mse:.6f} | "
                f"PSNR: {psnr:.2f} | SSIM: {ssim_val:.4f}"
            )
        else:
            title = (
                f"Sample {idx+1} | MAE: {mae:.4f} | MSE: {mse:.6f} | PSNR: {psnr:.2f}"
            )

        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

        save_path = os.path.join(vis_dir, f"sample_{idx+1:02d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"[Visualization] Saved {len(vis_images)} samples to {vis_dir}")
