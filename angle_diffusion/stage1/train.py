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

from .checkpoint import save_checkpoint
from .validation import validate



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
    criterion_l1 = CharbonnierLoss().to(device)
    criterion_l2 = MSELoss().to(device)
    print(f"[Stage1] Loss config: l1_weight={cfg.l1_weight}, l2_weight={cfg.l2_weight}")

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
            def get_weighted_loss(pred, target):
                return cfg.l1_weight * criterion_l1(pred, target) + cfg.l2_weight * criterion_l2(pred, target)

            l1 = get_weighted_loss(outputs[0], gt_img4)
            l2 = get_weighted_loss(outputs[1], gt_img2)
            l3 = get_weighted_loss(outputs[2], x0)
            loss_content = l1 + l2 + l3
            
            # Total loss (content only)
            loss = loss_content

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=float(np.mean(losses)))

        if (epoch + 1) % cfg.save_interval == 0:
            save_checkpoint(out_dir, f"epoch_{epoch+1:04d}.pth", model, model_le, optimizer, epoch + 1, best_val)

        if (epoch + 1) % cfg.val_interval == 0:
            metrics = validate(cfg, model, model_le, val_loader, device, epoch=epoch+1, out_dir=out_dir)
            val_ssim = metrics["val_ssim"]
            val_ssim_str = "None" if val_ssim is None else f"{val_ssim:.6f}"
            print(
                f"[Stage1] epoch={epoch+1} val_l1={metrics['val_l1']:.6f} "
                f"val_psnr={metrics['val_psnr']:.6f} val_ssim={val_ssim_str}"
            )
            current_val_l1: float = metrics["val_l1"]
            should_save = best_val is None or current_val_l1 < best_val
            if should_save:
                best_val = current_val_l1
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
