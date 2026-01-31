from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import runpy


@dataclass
class Stage1Config:
    # Data
    data_root: str
    train_split: str
    val_split: str
    max_samples: int | None

    # Training
    batch_size: int
    lr: float
    epochs: int
    num_workers: int
    save_interval: int
    val_interval: int
    seed: int
    val_ratio: float  # Validation dataset ratio (0.0-1.0)
    val_visualize: bool  # Whether to save visualizations during validation
    val_vis_samples: int  # Number of samples to visualize


    # Model - MIMO-UNet
    image_size: int
    num_res: int  # Number of residual blocks in MIMO-UNet
    in_channels: int  # Number of input channels (1 for CT)
    
    # Model - LatentEncoder
    n_feats: int  # Number of features in LatentEncoder
    n_encoder_res: int  # Number of residual blocks in encoder
    pixel_unshuffle_factor: int
    
    # Loss
    loss_type: str  # 'l1', 'l2', or 'weighted'
    l1_weight: float
    l2_weight: float
    
    # Output
    out_dir: str


def load_stage1_config(path: str | Path) -> Stage1Config:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".py"}:
        ns = runpy.run_path(str(path))
        if "cfg" not in ns or not isinstance(ns["cfg"], dict):
            raise ValueError(f"Config file {path} must define a dict named 'cfg'")
        cfg = ns["cfg"]
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "PyYAML is not installed. Use a .py config (recommended) or install pyyaml."
            ) from e
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config extension: {suffix} (use .py or .yaml)")

    return Stage1Config(
        data_root=cfg["data"]["root"],
        train_split=cfg["data"]["train_split"],
        val_split=cfg["data"]["val_split"],
        max_samples=cfg["data"].get("max_samples"),
        batch_size=cfg["training"]["batch_size"],
        lr=cfg["training"]["lr"],
        epochs=cfg["training"]["epochs"],
        num_workers=cfg["training"]["num_workers"],
        save_interval=cfg["training"]["save_interval"],
        val_interval=cfg["training"]["val_interval"],
        seed=cfg["training"].get("seed", 2023),
        val_ratio=cfg["training"].get("val_ratio", 1.0),
        val_visualize=cfg["training"].get("val_visualize", False),
        val_vis_samples=cfg["training"].get("val_vis_samples", 4),
        image_size=cfg["model"]["image_size"],
        num_res=cfg["model"].get("num_res", 20),
        in_channels=cfg["model"].get("in_channels", 1),
        n_feats=cfg["model"].get("n_feats", 64),
        n_encoder_res=cfg["model"].get("n_encoder_res", 6),
        pixel_unshuffle_factor=cfg["model"].get("pixel_unshuffle_factor", 4),
        loss_type=cfg["model"].get("loss_type", "l1"),
        l1_weight=cfg["model"].get("l1_weight", 1.0 if cfg["model"].get("loss_type", "l1") == "l1" else 0.0),
        l2_weight=cfg["model"].get("l2_weight", 1.0 if cfg["model"].get("loss_type", "l1") == "l2" else 0.0),
        out_dir=cfg["output"]["out_dir"],
    )
