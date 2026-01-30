from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from angle_diffusion.config import Stage1Config, load_stage1_config
from angle_diffusion.data.ct_dataset import CTDataset
from angle_diffusion.models.mimo_unet import build_MIMOUnet_net
from angle_diffusion.models.latent_encoder import LE_arch
from angle_diffusion.models.losses import CharbonnierLoss, MSELoss
from angle_diffusion.utils.misc import ensure_dir, save_yaml, seed_everything


def rfft(x: torch.Tensor, d: int) -> torch.Tensor:
    """FFT function compatible with both old and new PyTorch versions."""
    t = torch.fft.fft(x, dim=(-d))
    r = torch.stack((t.real, t.imag), -1)
    return r


def save_checkpoint(
    out_dir: str,
    name: str,
    model: torch.nn.Module,
    model_le: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float | None = None,
) -> str:
    path = os.path.join(out_dir, "checkpoints", name)
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "model_le": model_le.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val": best_val,
    }
    torch.save(payload, path)
    return path


@torch.no_grad()
def validate(
    cfg: Stage1Config,
    model: torch.nn.Module,
    model_le: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int | None = None,
    out_dir: str | None = None,
) -> float:
    """Validate using precomputed SART images from dataset."""
    model.eval()
    model_le.eval()
    losses = []
    
    # 可视化存储
    vis_images = []
    
    # 根据 val_ratio 计算需要使用的数据量
    total_batches = len(loader)
    num_batches = max(1, int(total_batches * cfg.val_ratio))
    
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
        losses.append(F.l1_loss(pred, x0).item())
        
        # 收集可视化样本
        if cfg.val_visualize and len(vis_images) < cfg.val_vis_samples:
            # 每个样本保存：原图、退化图、预测图
            for i in range(min(bs, cfg.val_vis_samples - len(vis_images))):
                vis_images.append({
                    'gt': x0[i].cpu(),
                    'degraded': x_t[i].cpu(),
                    'pred': pred[i].cpu(),
                })
    
    # 保存可视化结果
    if cfg.val_visualize and vis_images and epoch is not None and out_dir is not None:
        save_visualizations(out_dir, vis_images, epoch)
    
    return float(np.mean(losses)) if losses else 0.0


def save_visualizations(
    out_dir: str,
    vis_images: list[dict], 
    epoch: int, 
) -> None:
    """保存验证可视化结果"""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    vis_dir = os.path.join(out_dir, "logs", "visualizations", f"epoch_{epoch:04d}")
    ensure_dir(vis_dir)
    
    for idx, img_dict in enumerate(vis_images):
        gt = img_dict['gt'].squeeze().numpy()  # [H, W]
        degraded = img_dict['degraded'].squeeze().numpy()
        pred = img_dict['pred'].squeeze().numpy()
        
        # 创建对比图
        fig = plt.figure(figsize=(15, 5))
        gs = GridSpec(1, 3, figure=fig, wspace=0.3)
        
        # Ground Truth
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(gt, cmap='gray', vmin=0, vmax=1)
        ax1.set_title('Ground Truth', fontsize=12, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # SART Input
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(degraded, cmap='gray', vmin=0, vmax=1)
        ax2.set_title('SART Input', fontsize=12, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # Prediction
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(pred, cmap='gray', vmin=0, vmax=1)
        ax3.set_title('Prediction', fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        # 计算误差指标
        mae = np.mean(np.abs(pred - gt))
        mse = np.mean((pred - gt) ** 2)
        
        fig.suptitle(f'Sample {idx+1} | MAE: {mae:.4f} | MSE: {mse:.6f}', 
                     fontsize=14, fontweight='bold', y=0.98)
        
        save_path = os.path.join(vis_dir, f"sample_{idx+1:02d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"[Visualization] Saved {len(vis_images)} samples to {vis_dir}")


def train(cfg: Stage1Config, resume: str | None = None) -> None:
    from datetime import datetime
    
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建时间戳子目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out_dir = ensure_dir(cfg.out_dir)
    out_dir = ensure_dir(os.path.join(base_out_dir, timestamp))
    ensure_dir(os.path.join(out_dir, "checkpoints"))
    ensure_dir(os.path.join(out_dir, "logs"))
    save_yaml(cfg, os.path.join(out_dir, "stage1_config_resolved.yaml"))

    print(f"[Stage1] device={device}")
    print(f"[Stage1] out_dir={out_dir}")

    # Initialize MIMO-UNet model
    model = build_MIMOUnet_net(
        "MIMOUNetBlurDM",
        num_res=cfg.num_res,
        in_channels=cfg.in_channels,
    ).to(device)

    # Initialize LatentEncoder
    model_le = LE_arch(
        n_feats=cfg.n_feats,
        n_encoder_res=cfg.n_encoder_res,
        in_channels=cfg.in_channels,
        pixel_unshuffle_factor=cfg.pixel_unshuffle_factor,
    ).to(device)

    # Optimizer for both models
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(model_le.parameters()), 
        lr=cfg.lr
    )

    # Loss criterion
    if cfg.loss_type == "l1":
        criterion = CharbonnierLoss().to(device)
        print(f"[Stage1] Using L1 loss (CharbonnierLoss)")
    elif cfg.loss_type == "l2":
        criterion = MSELoss().to(device)
        print(f"[Stage1] Using L2 loss (MSELoss)")
    else:
        raise ValueError(f"Unsupported loss_type: {cfg.loss_type}. Choose 'l1' or 'l2'.")

    train_ds = CTDataset(cfg.data_root, split=cfg.train_split, max_samples=cfg.max_samples)
    val_ds = CTDataset(cfg.data_root, split=cfg.val_split, max_samples=cfg.max_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    start_epoch = 0
    best_val = None
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model_le.load_state_dict(ckpt["model_le"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0))
        best_val = ckpt.get("best_val")
        print(f"[Stage1] resumed from {resume} @ epoch={start_epoch}")

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        model_le.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Stage1 Ep {epoch+1}/{cfg.epochs}")

        for x0, x_t in pbar:
            x0 = x0.to(device)  # [B,1,H,W] GT
            x_t = x_t.to(device)  # [B,1,H,W] SART input

            # Extract latent prior from SART and GT images
            z_pred = model_le(x_t, x0)
            
            # Get multi-scale predictions [1/4x, 1/2x, 1x]
            outputs = model(x_t, z_pred)
            outputs = [output.clamp(-0.5, 0.5) for output in outputs]
            
            # Multi-scale ground truth
            gt_img2 = F.interpolate(x0, scale_factor=0.5, mode='bilinear')
            gt_img4 = F.interpolate(x0, scale_factor=0.25, mode='bilinear')
            
            # Content loss at each scale
            l1 = criterion(outputs[0], gt_img4)
            l2 = criterion(outputs[1], gt_img2)
            l3 = criterion(outputs[2], x0)
            loss_content = l1 + l2 + l3
            
            # FFT loss at each scale
            label_fft1 = rfft(gt_img4, 2)
            pred_fft1 = rfft(outputs[0], 2)
            label_fft2 = rfft(gt_img2, 2)
            pred_fft2 = rfft(outputs[1], 2)
            label_fft3 = rfft(x0, 2)
            pred_fft3 = rfft(outputs[2], 2)
            
            f1 = criterion(pred_fft1, label_fft1)
            f2 = criterion(pred_fft2, label_fft2)
            f3 = criterion(pred_fft3, label_fft3)
            loss_fft = f1 + f2 + f3
            
            # Total loss
            loss = loss_content + 0.1 * loss_fft

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=float(np.mean(losses)))

        if (epoch + 1) % cfg.save_interval == 0:
            save_checkpoint(out_dir, f"epoch_{epoch+1:04d}.pth", model, model_le, optimizer, epoch + 1, best_val)

        if (epoch + 1) % cfg.val_interval == 0:
            val_loss = validate(cfg, model, model_le, val_loader, device, epoch=epoch+1, out_dir=out_dir)
            print(f"[Stage1] epoch={epoch+1} val_l1={val_loss:.6f}")
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                save_checkpoint(out_dir, "best.pth", model, model_le, optimizer, epoch + 1, best_val)

    save_checkpoint(out_dir, "last.pth", model, model_le, optimizer, cfg.epochs, best_val)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--resume", default=None, type=str)
    args = parser.parse_args()

    cfg = load_stage1_config(Path(args.config))
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()

