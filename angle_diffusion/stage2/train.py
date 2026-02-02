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
from angle_diffusion.data.ct_dataset import CTAngleDegradeDataset
from angle_diffusion.models.latent_encoder import LE_arch
from angle_diffusion.models.angle_encoder import AngleEncoder
from angle_diffusion.models.angle_dm import AngleDM
from angle_diffusion.physics.ct_physics import CT_Physics
from angle_diffusion.utils.misc import ensure_dir, save_yaml, seed_everything


def save_checkpoint(
    out_dir: str,
    name: str,
    be: torch.nn.Module,
    dm: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float | None = None,
) -> str:
    path = os.path.join(out_dir, "checkpoints", name)
    payload = {
        "epoch": epoch,
        "be": be.state_dict(),
        "dm": dm.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val": best_val,
    }
    torch.save(payload, path)
    return path


@torch.no_grad()
def validate(
    cfg: Stage2Config,
    teacher: torch.nn.Module,
    be: torch.nn.Module,
    dm: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int | None = None,
    out_dir: str | None = None,
) -> dict[str, float]:
    """Validate by latent L1: ||Z0^B - Z^S||_1."""
    teacher.eval()
    be.eval()
    dm.eval()

    losses = []

    total_batches = len(loader)
    num_batches = max(1, int(total_batches * cfg.val_ratio))

    for batch_idx, (x0, x_deg, t) in enumerate(loader):
        if batch_idx >= num_batches:
            break

        x0 = x0.to(device)
        x_deg = x_deg.to(device)
        if isinstance(t, torch.Tensor):
            t_tensor = t.to(device)
        else:
            t_tensor = torch.tensor(t, device=device)

        # Teacher sharp prior (from Stage1 SE): Z^S = LE_arch(x_deg, x0)
        z_s = teacher(x_deg, x0)

        # Condition latent from degraded only: Z^B
        z_b = be(x_deg)

        # One-step forward: Z_T = Z_B + beta_bar[T]*eps
        z0_pred = dm.sample(z_b, degrade_level=t_tensor)

        loss = F.l1_loss(z0_pred, z_s)
        losses.append(float(loss.detach().cpu().item()))

    val_l1 = float(np.mean(losses)) if losses else 0.0

    if epoch is not None and out_dir is not None:
        logs_dir = ensure_dir(os.path.join(out_dir, "logs"))
        metrics_path = os.path.join(logs_dir, "metrics.csv")
        is_new = not os.path.exists(metrics_path)
        with open(metrics_path, "a", encoding="utf-8") as f:
            if is_new:
                f.write("epoch,val_l1\n")
            f.write(f"{epoch},{val_l1:.6f}\n")

    return {"val_l1": val_l1}


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
    ct_physics = CT_Physics(
        focus_table_path=cfg.focus_table_path,
        sart_iterations=cfg.sart_iterations,
        preset=cfg.preset,
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

    train_ds = CTAngleDegradeDataset(
        root_dir=cfg.data_root,
        split=cfg.train_split,
        ct_physics=ct_physics,
        t_min=cfg.degrade_t_min,
        t_max=cfg.degrade_t_max,
        t_values=cfg.degrade_t_values,
        max_samples=cfg.max_samples,
        cache_dir=train_cache,
        cache_clip01=True,
    )
    # Validation uses a fixed degradation level for stability.
    val_ds = CTAngleDegradeDataset(
        root_dir=cfg.data_root,
        split=cfg.val_split,
        ct_physics=ct_physics,
        t_min=cfg.degrade_t_val,
        t_max=cfg.degrade_t_val,
        t_values=None,
        max_samples=cfg.max_samples,
        cache_dir=val_cache,
        cache_clip01=True,
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

        for x0, x_deg, t in pbar:
            x0 = x0.to(device)  # [-0.5,0.5]
            x_deg = x_deg.to(device)
            if isinstance(t, torch.Tensor):
                t_tensor = t.to(device)
            else:
                t_tensor = torch.tensor(t, device=device)

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
                val_loader,
                device,
                epoch=epoch + 1,
                out_dir=out_dir,
            )
            print(f"[Stage2] epoch={epoch + 1} val_l1={metrics['val_l1']:.6f}")
            if best_val is None or metrics["val_l1"] < best_val:
                best_val = metrics["val_l1"]
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
