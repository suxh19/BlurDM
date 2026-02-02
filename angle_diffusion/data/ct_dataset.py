"""
CT dataset (GT + optional SART pairs).

Copied from code_diffusion/cold_diffuson/utils/dataset.py with minimal changes.
"""

from __future__ import annotations

from collections.abc import Callable
import os
import random
from pathlib import Path

import numpy as np
import torch
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

        gt_paths = sorted(gt_dir.glob("*.npy"))
        if len(gt_paths) == 0:
            raise ValueError(f"No NPY files found in {gt_dir}")

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

        # Load GT from npy (already normalized to [0, 1])
        gt_np = np.load(gt_path).astype(np.float32)
        # Convert to [-0.5, 0.5] range (model requirement)
        gt_np = gt_np - 0.5
        if gt_np.ndim == 2:
            gt_tensor = torch.from_numpy(gt_np).unsqueeze(0)  # [1, H, W]
        else:
            gt_tensor = torch.from_numpy(gt_np)

        # Load SART from npy (already normalized to [0, 1])
        sart_np = np.load(sart_path).astype(np.float32)
        # Convert to [-0.5, 0.5] range (model requirement)
        sart_np = sart_np - 0.5
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


class CTGroundTruthDataset(Dataset):
    """Ground-truth-only dataset.

    It loads GT npy files under:
        {root_dir}/{split}/ground_truth/*.npy

    Values are expected to be normalized to [0, 1]. This dataset returns tensors
    in the *model range* [-0.5, 0.5] (consistent with Stage1).
    """

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
        if not gt_dir.exists():
            raise ValueError(f"Ground truth directory not found: {gt_dir}")

        self.gt_paths = sorted(gt_dir.glob("*.npy"))
        if len(self.gt_paths) == 0:
            raise ValueError(f"No NPY files found in {gt_dir}")

        if max_samples is not None and max_samples > 0:
            self.gt_paths = self.gt_paths[:max_samples]

        print(
            f"[CTGroundTruthDataset] Loaded {len(self.gt_paths)} GT files from {split}"
        )

    def __len__(self) -> int:
        return len(self.gt_paths)

    def __getitem__(self, idx: int):
        gt_path = self.gt_paths[idx]
        gt_np = np.load(gt_path).astype(np.float32)  # [0,1]
        gt_np_shift = gt_np - 0.5
        if gt_np_shift.ndim == 2:
            gt_tensor = torch.from_numpy(gt_np_shift).unsqueeze(0)
        else:
            gt_tensor = torch.from_numpy(gt_np_shift)

        if self.transform is not None:
            gt_tensor = self.transform(gt_tensor)

        if self.return_path:
            return gt_tensor, str(gt_path)
        return gt_tensor


class CTAngleDegradeDataset(Dataset):
    """On-the-fly CT degradation dataset using CT_Physics.

    For each GT image x0, we sample a degradation level t in [t_min, t_max]
    and generate x_deg = CT_Physics.degrade(x0, t).

    Assumption ("Case A"):
        The physics operator already returns reconstructions in a consistent
        intensity scale, so we *do not* apply any additional alpha/normalization
        factor beyond the dataset's [-0.5, 0.5] shift.

    Caching:
        If cache_dir is provided, degraded results will be cached as .npy files
        to avoid recomputing CT projections/reconstructions every epoch.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        ct_physics,
        t_min: int = 100,
        t_max: int = 100,
        t_values: list[int] | None = None,
        transform: Callable | None = None,
        return_path: bool = False,
        max_samples: int | None = None,
        cache_dir: str | None = None,
        cache_clip01: bool = True,
    ):
        if not (0 <= t_min <= 100 and 0 <= t_max <= 100 and t_min <= t_max):
            raise ValueError(f"Invalid t range: [{t_min}, {t_max}]")

        if t_values is not None:
            if not isinstance(t_values, list) or not t_values:
                raise ValueError("t_values must be a non-empty list")
            vals = [int(v) for v in t_values]
            for v in vals:
                if not (0 <= v <= 100):
                    raise ValueError("t_values entries must be in [0,100]")
            if t_min not in vals or t_max not in vals:
                raise ValueError("t_values must include t_min and t_max")
            t_values = sorted(set(vals))

        self.root_dir = Path(root_dir)
        self.split = split
        self.ct_physics = ct_physics
        self.t_min = int(t_min)
        self.t_max = int(t_max)
        self.t_values = t_values
        self.transform = transform
        self.return_path = return_path
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_clip01 = bool(cache_clip01)

        gt_dir = self.root_dir / split / "ground_truth"
        if not gt_dir.exists():
            raise ValueError(f"Ground truth directory not found: {gt_dir}")

        self.gt_paths = sorted(gt_dir.glob("*.npy"))
        if len(self.gt_paths) == 0:
            raise ValueError(f"No NPY files found in {gt_dir}")

        if max_samples is not None and max_samples > 0:
            self.gt_paths = self.gt_paths[:max_samples]

        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

        if self.t_values is None:
            t_desc = f"t_range=[{self.t_min},{self.t_max}]"
        else:
            t_desc = f"t_values={self.t_values}"
        print(
            f"[CTAngleDegradeDataset] Loaded {len(self.gt_paths)} GT files from {split} | "
            f"{t_desc} | cache={self.cache_dir is not None}"
        )

    def __len__(self) -> int:
        return len(self.gt_paths)

    def _cache_path(self, gt_path: Path, t: int) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / f"{gt_path.stem}_t{t:03d}.npy"

    def __getitem__(self, idx: int):
        gt_path = self.gt_paths[idx]
        if self.t_values is None:
            t = random.randint(self.t_min, self.t_max)
        else:
            t = random.choice(self.t_values)

        gt_np = np.load(gt_path).astype(np.float32)  # [0,1]

        # Degrade using physics operator (expects [0,1] scale in our dataset).
        if self.cache_dir is not None:
            cache_path = self._cache_path(gt_path, t)
            if cache_path.exists():
                deg_np = np.load(cache_path).astype(np.float32)
            else:
                deg_np = self.ct_physics.degrade(gt_np, t).astype(np.float32)
                if self.cache_clip01:
                    deg_np = np.clip(deg_np, 0.0, 1.0)
                np.save(cache_path, deg_np)
        else:
            deg_np = self.ct_physics.degrade(gt_np, t).astype(np.float32)
            if self.cache_clip01:
                deg_np = np.clip(deg_np, 0.0, 1.0)

        # Shift to model range [-0.5, 0.5]
        gt_shift = gt_np - 0.5
        deg_shift = deg_np - 0.5

        if gt_shift.ndim == 2:
            gt_tensor = torch.from_numpy(gt_shift).unsqueeze(0)
        else:
            gt_tensor = torch.from_numpy(gt_shift)
        if deg_shift.ndim == 2:
            deg_tensor = torch.from_numpy(deg_shift).unsqueeze(0)
        else:
            deg_tensor = torch.from_numpy(deg_shift)

        if self.transform is not None:
            gt_tensor = self.transform(gt_tensor)
            deg_tensor = self.transform(deg_tensor)

        if self.return_path:
            return gt_tensor, deg_tensor, t, str(gt_path)
        return gt_tensor, deg_tensor, t
