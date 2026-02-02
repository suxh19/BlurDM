from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from angle_diffusion.config import Stage2Config, load_stage2_config
from angle_diffusion.data.ct_dataset import CTGroundTruthDataset
from angle_diffusion.models.latent_encoder import LE_arch
from angle_diffusion.models.mimo_unet import build_MIMOUnet_net
from angle_diffusion.models.angle_encoder import AngleEncoder
from angle_diffusion.models.angle_dm import AngleDM
from angle_diffusion.physics.ct_physics_batch import CT_PhysicsBatch
from angle_diffusion.utils.misc import ensure_dir, save_yaml, seed_everything

from .checkpoint import save_checkpoint
from .degradation import sample_degrade_level, degrade_batch_with_cache
from .validation import validate


def train(cfg: Stage2Config, resume: str | None = None) -> None:
    from datetime import datetime

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out_dir = ensure_dir(cfg.out_dir)
    out_dir = ensure_dir(os.path.join(base_out_dir, timestamp))
    ensure_dir(os.path.join(out_dir, "checkpoints"))
    ensure_dir(os.path.join(out_dir, "logs"))
    save_yaml(cfg, os.path.join(out_dir, "stage2_config_resolved.yaml"))

    print(f"[Stage2] device={device}")
    print(f"[Stage2] out_dir={out_dir}")

    # ------------------------------------------------------------
    # Physics operator
    # ------------------------------------------------------------
    ct_physics = CT_PhysicsBatch(
        focus_table_path=cfg.focus_table_path,
        sart_iterations=cfg.sart_iterations,
        preset=cfg.preset,
        fp_algorithm=cfg.fp_algorithm,
        sirt_algorithm=cfg.sirt_algorithm,
        batch_size=cfg.physics_batch_size,
        num_workers=cfg.physics_num_workers,
    )

    # ------------------------------------------------------------
    # Teacher (Stage1) latent encoder
    # ------------------------------------------------------------
    teacher = LE_arch(
        n_feats=cfg.n_feats,
        n_encoder_res=cfg.n_encoder_res,
        in_channels=cfg.in_channels,
        pixel_unshuffle_factor=cfg.pixel_unshuffle_factor,
    ).to(device)
    teacher.eval()

    ckpt = torch.load(cfg.stage1_ckpt, map_location=device, weights_only=False)
    if "model_le" not in ckpt:
        raise ValueError(
            "Stage1 checkpoint must contain key 'model_le'. "
            "Please point stage1.ckpt to Stage1 'best.pth' or 'last.pth'."
        )
    teacher.load_state_dict(ckpt["model_le"], strict=True)
    for p in teacher.parameters():
        p.requires_grad_(False)

    decoder = None
    if cfg.val_visualize:
        if "model" not in ckpt:
            raise ValueError(
                "Stage1 checkpoint must contain key 'model' for visualization. "
                "Please point stage1.ckpt to Stage1 'best.pth' or 'last.pth'."
            )
        decoder = build_MIMOUnet_net(
            "MIMOUNetBlurDM",
            num_res=cfg.decoder_num_res,
            in_channels=cfg.in_channels,
        ).to(device)
        decoder.load_state_dict(ckpt["model"], strict=True)
        decoder.eval()
        for p in decoder.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------
    # Trainable BE + AngleDM
    # ------------------------------------------------------------
    be = AngleEncoder(
        n_feats=cfg.n_feats,
        n_encoder_res=cfg.n_encoder_res,
        in_channels=cfg.in_channels,
        pixel_unshuffle_factor=cfg.pixel_unshuffle_factor,
    ).to(device)

    latent_dim = cfg.n_feats * 4
    dm = AngleDM(
        latent_dim=latent_dim,
        T=cfg.T,
        beta_max=cfg.beta_max,
        time_embed_dim=cfg.time_embed_dim,
        hidden_dim=cfg.hidden_dim,
        use_degrade_level_cond=cfg.use_degrade_level_cond,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(be.parameters()) + list(dm.parameters()), lr=cfg.lr
    )

    # ------------------------------------------------------------
    # Datasets / loaders
    # ------------------------------------------------------------
    train_cache = None
    val_cache = None
    if cfg.cache_dir:
        train_cache = os.path.join(cfg.cache_dir, "train")
        val_cache = os.path.join(cfg.cache_dir, "val")

    train_ds = CTGroundTruthDataset(
        root_dir=cfg.data_root,
        split=cfg.train_split,
        return_path=True,
        max_samples=cfg.max_samples,
    )
    val_ds = CTGroundTruthDataset(
        root_dir=cfg.data_root,
        split=cfg.val_split,
        return_path=True,
        max_samples=cfg.max_samples,
    )

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
        ckpt2 = torch.load(resume, map_location=device, weights_only=False)
        be.load_state_dict(ckpt2["be"], strict=True)
        dm.load_state_dict(ckpt2["dm"], strict=True)
        optimizer.load_state_dict(ckpt2["optimizer"])
        start_epoch = int(ckpt2.get("epoch", 0))
        best_val = ckpt2.get("best_val")
        print(f"[Stage2] resumed from {resume} @ epoch={start_epoch}")

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    for epoch in range(start_epoch, cfg.epochs):
        be.train()
        dm.train()

        losses = []
        pbar = tqdm(train_loader, desc=f"Stage2 Ep {epoch + 1}/{cfg.epochs}")

        for x0, gt_paths in pbar:
            if isinstance(gt_paths, (list, tuple)):
                paths = [str(p) for p in gt_paths]
            else:
                paths = [str(gt_paths)]

            t_level = sample_degrade_level(cfg)
            gt_01 = (x0 + 0.5).clamp(0.0, 1.0)
            gt_01_np = gt_01.numpy().astype(np.float32, copy=False)
            deg_np = degrade_batch_with_cache(
                ct_physics,
                gt_01_np,
                paths,
                t_level,
                train_cache,
                clip_01=True,
            )
            deg_shift = deg_np - 0.5
            x0 = x0.to(device)  # [-0.5,0.5]
            x_deg = torch.from_numpy(deg_shift).to(device)
            t_tensor = torch.full((x0.shape[0],), float(t_level), device=device)

            # Teacher Z^S
            with torch.no_grad():
                z_s = teacher(x_deg, x0)

            # Z^B from degraded only
            z_b = be(x_deg)

            # One-step forward
            eps = torch.randn_like(z_b)
            z_T = z_b + dm.beta_bar[dm.T] * eps

            # Reverse to predict Z0
            z0_pred = dm.reverse_from(z_T, z_b, degrade_level=t_tensor)

            loss = F.l1_loss(z0_pred, z_s)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.detach().cpu().item()))
            pbar.set_postfix(loss=float(np.mean(losses)))

        if (epoch + 1) % cfg.save_interval == 0:
            save_checkpoint(
                out_dir,
                f"epoch_{epoch + 1:04d}.pth",
                be,
                dm,
                optimizer,
                epoch + 1,
                best_val,
            )

        if (epoch + 1) % cfg.val_interval == 0:
            metrics = validate(
                cfg,
                teacher,
                be,
                dm,
                decoder,
                val_loader,
                ct_physics,
                val_cache,
                device,
                epoch=epoch + 1,
                out_dir=out_dir,
            )
            print(f"[Stage2] epoch={epoch + 1} val_l1={metrics['val_l1']:.6f}")
            current_val_l1: float = metrics["val_l1"]
            should_save = best_val is None or current_val_l1 < best_val
            if should_save:
                best_val = current_val_l1
                save_checkpoint(
                    out_dir, "best.pth", be, dm, optimizer, epoch + 1, best_val
                )

    save_checkpoint(out_dir, "last.pth", be, dm, optimizer, cfg.epochs, best_val)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--resume", default=None, type=str)
    args = parser.parse_args()

    cfg = load_stage2_config(Path(args.config))
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
