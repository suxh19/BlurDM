"""
CT dataset (GT + optional SART pairs).

Copied from code_diffusion/cold_diffuson/utils/dataset.py with minimal changes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CTDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Callable | None = None,
        return_path: bool = False,
        max_samples: int | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.return_path = return_path

        gt_dir = self.root_dir / split / "ground_truth"
        sart_dir = self.root_dir / split / "sart"

        if not gt_dir.exists():
            raise ValueError(f"Ground truth directory not found: {gt_dir}")
        if not sart_dir.exists():
            raise ValueError(f"SART directory not found: {sart_dir}")

        gt_paths = sorted(gt_dir.glob("*.png"))
        if len(gt_paths) == 0:
            raise ValueError(f"No PNG images found in {gt_dir}")

        self.pairs: list[tuple[Path, Path]] = []
        for gt_path in gt_paths:
            stem = gt_path.stem
            sart_path = sart_dir / f"{stem}.npy"
            if sart_path.exists():
                self.pairs.append((gt_path, sart_path))

        if len(self.pairs) == 0:
            raise ValueError(f"No matching GT-SART pairs found in {split}")

        if max_samples is not None and max_samples > 0:
            self.pairs = self.pairs[:max_samples]

        print(f"[CTDataset] Loaded {len(self.pairs)} GT-SART pairs from {split}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        gt_path, sart_path = self.pairs[idx]

        gt_img = Image.open(gt_path).convert("L")
        gt_np = np.array(gt_img).astype(np.float32) / 255.0
        gt_tensor = torch.from_numpy(gt_np).unsqueeze(0)  # [1, H, W]

        sart_np = np.load(sart_path).astype(np.float32)
        if sart_np.ndim == 2:
            sart_tensor = torch.from_numpy(sart_np).unsqueeze(0)
        else:
            sart_tensor = torch.from_numpy(sart_np)

        if self.transform is not None:
            gt_tensor = self.transform(gt_tensor)
            sart_tensor = self.transform(sart_tensor)

        if self.return_path:
            return gt_tensor, sart_tensor, str(gt_path), str(sart_path)

        return gt_tensor, sart_tensor

