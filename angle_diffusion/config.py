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


@dataclass
class Stage2Config:
    # Data
    data_root: str
    train_split: str
    val_split: str
    max_samples: int | None
    cache_dir: str | None

    # Stage1 teacher checkpoint (provides LE_arch weights)
    stage1_ckpt: str

    # Physics (CT degradation)
    focus_table_path: str
    sart_iterations: int
    preset: str
    degrade_t_min: int
    degrade_t_max: int
    degrade_t_val: int
    degrade_t_num_levels: int | None
    degrade_t_values: list[int] | None

    # Training
    batch_size: int
    lr: float
    epochs: int
    num_workers: int
    save_interval: int
    val_interval: int
    seed: int
    val_ratio: float

    # Model - encoder (BE)
    in_channels: int
    n_feats: int
    n_encoder_res: int
    pixel_unshuffle_factor: int

    # Model - diffusion
    T: int
    beta_max: float
    time_embed_dim: int
    hidden_dim: int
    use_degrade_level_cond: bool

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

    model_cfg = cfg["model"]
    loss_type = model_cfg.get("loss_type", "l1")
    l1_weight = model_cfg.get("l1_weight")
    l2_weight = model_cfg.get("l2_weight")
    if loss_type == "weighted":
        if l1_weight is None or l2_weight is None:
            raise ValueError(
                "loss_type 'weighted' requires explicit l1_weight and l2_weight in config"
            )
    elif loss_type == "l1":
        if l1_weight is None:
            l1_weight = 1.0
        if l2_weight is None:
            l2_weight = 0.0
    elif loss_type == "l2":
        if l1_weight is None:
            l1_weight = 0.0
        if l2_weight is None:
            l2_weight = 1.0
    else:
        raise ValueError(
            f"Unsupported loss_type: {loss_type!r} (use 'l1', 'l2', or 'weighted')"
        )

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
        image_size=model_cfg["image_size"],
        num_res=model_cfg.get("num_res", 20),
        in_channels=model_cfg.get("in_channels", 1),
        n_feats=model_cfg.get("n_feats", 64),
        n_encoder_res=model_cfg.get("n_encoder_res", 6),
        pixel_unshuffle_factor=model_cfg.get("pixel_unshuffle_factor", 4),
        loss_type=loss_type,
        l1_weight=float(l1_weight) if l1_weight is not None else 0.0,
        l2_weight=float(l2_weight) if l2_weight is not None else 0.0,
        out_dir=cfg["output"]["out_dir"],
    )


def load_stage2_config(path: str | Path) -> Stage2Config:
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

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    phys_cfg = cfg["physics"]
    deg_cfg = cfg["degradation"]

    t_min_raw = deg_cfg.get("t_min")
    t_max_raw = deg_cfg.get("t_max")
    t_val_raw = deg_cfg.get("t_val")
    t_num_levels_raw = deg_cfg.get("t_num_levels")
    t_values_raw = deg_cfg.get("t_values")

    if t_num_levels_raw is not None and t_values_raw is not None:
        raise ValueError(
            "Specify only one of degradation.t_num_levels or degradation.t_values"
        )

    degrade_t_values: list[int] | None = None
    degrade_t_num_levels: int | None = None

    if t_values_raw is not None:
        if not isinstance(t_values_raw, list) or not t_values_raw:
            raise ValueError("degradation.t_values must be a non-empty list")
        vals = [int(v) for v in t_values_raw]
        for v in vals:
            if not (0 <= v <= 100):
                raise ValueError("degradation.t_values entries must be in [0,100]")
        degrade_t_values = sorted(set(vals))
        degrade_t_num_levels = len(degrade_t_values)
        degrade_t_min = min(degrade_t_values)
        degrade_t_max = max(degrade_t_values)
        if t_min_raw is not None and int(t_min_raw) != degrade_t_min:
            raise ValueError("degradation.t_min must match min(t_values)")
        if t_max_raw is not None and int(t_max_raw) != degrade_t_max:
            raise ValueError("degradation.t_max must match max(t_values)")
    elif t_num_levels_raw is not None:
        degrade_t_num_levels = int(t_num_levels_raw)
        if degrade_t_num_levels <= 0:
            raise ValueError("degradation.t_num_levels must be >= 1")

        degrade_t_min = int(t_min_raw) if t_min_raw is not None else 0
        degrade_t_max = int(t_max_raw) if t_max_raw is not None else 100
        if not (
            0 <= degrade_t_min <= 100
            and 0 <= degrade_t_max <= 100
            and degrade_t_min <= degrade_t_max
        ):
            raise ValueError("degradation.t_min/t_max must satisfy 0<=min<=max<=100")

        if degrade_t_num_levels == 1:
            if degrade_t_min != degrade_t_max:
                raise ValueError("degradation.t_num_levels=1 requires t_min == t_max")
            degrade_t_values = [degrade_t_min]
        else:
            span = degrade_t_max - degrade_t_min
            den = degrade_t_num_levels - 1
            if span < den:
                raise ValueError(
                    "degradation.t_num_levels is too large for integer range; require (t_max - t_min) >= (t_num_levels - 1)"
                )
            vals = [
                degrade_t_min + (i * span + (den // 2)) // den
                for i in range(degrade_t_num_levels)
            ]
            vals[0] = degrade_t_min
            vals[-1] = degrade_t_max
            if len(set(vals)) != degrade_t_num_levels:
                raise ValueError(
                    "degradation.t_num_levels produced duplicate t values; use a smaller value or specify degradation.t_values"
                )
            degrade_t_values = vals
    else:
        degrade_t_min = int(t_min_raw) if t_min_raw is not None else 100
        degrade_t_max = int(t_max_raw) if t_max_raw is not None else degrade_t_min
        if not (
            0 <= degrade_t_min <= 100
            and 0 <= degrade_t_max <= 100
            and degrade_t_min <= degrade_t_max
        ):
            raise ValueError("degradation.t_min/t_max must satisfy 0<=min<=max<=100")

    degrade_t_val = int(t_val_raw) if t_val_raw is not None else degrade_t_max
    if not (0 <= degrade_t_val <= 100):
        raise ValueError("degradation.t_val must be in [0,100]")
    if not (degrade_t_min <= degrade_t_val <= degrade_t_max):
        raise ValueError("degradation.t_val must be within [t_min, t_max]")

    return Stage2Config(
        data_root=data_cfg["root"],
        train_split=data_cfg.get("train_split", "train"),
        val_split=data_cfg.get("val_split", "test"),
        max_samples=data_cfg.get("max_samples"),
        cache_dir=data_cfg.get("cache_dir"),
        stage1_ckpt=cfg["stage1"]["ckpt"],
        focus_table_path=phys_cfg["focus_table_path"],
        sart_iterations=int(phys_cfg.get("sart_iterations", 10)),
        preset=str(phys_cfg.get("preset", "standard_512")),
        degrade_t_min=degrade_t_min,
        degrade_t_max=degrade_t_max,
        degrade_t_val=degrade_t_val,
        degrade_t_num_levels=degrade_t_num_levels,
        degrade_t_values=degrade_t_values,
        batch_size=int(train_cfg["batch_size"]),
        lr=float(train_cfg["lr"]),
        epochs=int(train_cfg["epochs"]),
        num_workers=int(train_cfg.get("num_workers", 4)),
        save_interval=int(train_cfg.get("save_interval", 10)),
        val_interval=int(train_cfg.get("val_interval", 5)),
        seed=int(train_cfg.get("seed", 2023)),
        val_ratio=float(train_cfg.get("val_ratio", 1.0)),
        in_channels=int(model_cfg.get("in_channels", 1)),
        n_feats=int(model_cfg.get("n_feats", 64)),
        n_encoder_res=int(model_cfg.get("n_encoder_res", 6)),
        pixel_unshuffle_factor=int(model_cfg.get("pixel_unshuffle_factor", 4)),
        T=int(model_cfg.get("T", 5)),
        beta_max=float(model_cfg.get("beta_max", 0.02)),
        time_embed_dim=int(model_cfg.get("time_embed_dim", 128)),
        hidden_dim=int(model_cfg.get("hidden_dim", 512)),
        use_degrade_level_cond=bool(model_cfg.get("use_degrade_level_cond", True)),
        out_dir=cfg["output"]["out_dir"],
    )
