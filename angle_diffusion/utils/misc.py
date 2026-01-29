from __future__ import annotations

import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str | Path) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def save_yaml(obj, path: str | Path) -> None:
    data = asdict(obj)
    path = Path(path)
    # Prefer YAML if available, otherwise fall back to JSON for zero-dep saving.
    try:
        import yaml  # type: ignore

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
    except ModuleNotFoundError:
        import json

        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def to_batch_numpy(x: torch.Tensor) -> npt.NDArray[np.floating[Any]]:
    arr: npt.NDArray[np.floating[Any]] = x.detach().cpu().numpy().astype(np.float32)
    # [B, 1, H, W] -> [B, H, W]
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr

