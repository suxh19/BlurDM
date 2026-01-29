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

    # Physics / degradation
    focus_table_path: str
    max_t: int
    t_min: int
    t_max: int
    sart_iterations: int
    config_preset: str

    # Model
    image_size: int
    base_channels: int
    channel_mults: list[int]
    num_res_blocks: int

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
        focus_table_path=cfg["physics"]["focus_table_path"],
        max_t=cfg["physics"]["max_t"],
        t_min=cfg["physics"].get("t_min", 0),
        t_max=cfg["physics"].get("t_max", cfg["physics"]["max_t"]),
        sart_iterations=cfg["physics"]["sart_iterations"],
        config_preset=cfg["physics"]["config_preset"],
        image_size=cfg["model"]["image_size"],
        base_channels=cfg["model"].get("base_channels", 64),
        channel_mults=list(cfg["model"].get("channel_mults", [1, 2, 4, 8])),
        num_res_blocks=cfg["model"].get("num_res_blocks", 2),
        out_dir=cfg["output"]["out_dir"],
    )
