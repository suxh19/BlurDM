"""
CT Physics degradation operator.

Copied from code_diffusion/cold_diffuson/ct_physics.py to keep this package
self-contained.
"""

from __future__ import annotations

import argparse
import os
import numpy as np
from pathlib import Path

from astra_package.ct.unified.projector import Projector3D
from astra_package.ct.unified.reconstructor import Reconstructor3D
from astra_package.ct.config.config import load_fromnpz


class CT_Physics:
    def __init__(
        self,
        focus_table_path: str,
        sart_iterations: int = 10,
        preset: str = "standard_512",
    ):
        self.focus_table_path = focus_table_path
        self.sart_iterations = sart_iterations
        self.preset = preset

        # 简化路径处理：相对路径相对于项目根目录，绝对路径直接使用
        path = Path(focus_table_path)
        if not path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            path = project_root / path
        
        if not path.exists():
            raise FileNotFoundError(f"focus table 不存在: {path}")
        
        data = np.load(path)

        self.ns = data["ns"].astype(int)

        print(
            f"[CT_Physics] 初始化完成 | SART={sart_iterations} | 源范围: {self.ns.min()}~{self.ns.max()}"
        )

    def degrade(self, image: np.ndarray, t: int) -> np.ndarray:
        if image.ndim == 2:
            image = image[None, ...]
        if t == 0:
            result = image.copy()
            if result.shape[0] == 1:
                result = result[0]
            return result

        if not 1 <= t <= 100:
            raise ValueError(f"t 必须在 [0, 100] 范围内，实际: {t}")

        table_index = 100 - t
        num_sources = int(self.ns[table_index])

        config = load_fromnpz(self.focus_table_path, num_sources, self.preset)
        config["reconstruction"]["SART"]["sart_iterations"] = self.sart_iterations

        projector = Projector3D(config)
        reconstructor = Reconstructor3D(config)

        sinogram = projector.project(image, rearrange=False)
        result = reconstructor.reconstruct(sinogram)
        if result.shape[0] == 1:
            result = result[0]
        return result

    def degrade_batch(self, images: np.ndarray, t: int) -> np.ndarray:
        if images.ndim == 2:
            return self.degrade(images, t)
        results = [self.degrade(img, t) for img in images]
        return np.stack(results, axis=0)

    def t_to_num_sources(self, t: int) -> int:
        if t == 0:
            return -1
        if not 1 <= t <= 100:
            raise ValueError(f"t 必须在 [0, 100] 范围内，实际: {t}")
        table_index = 100 - t
        return int(self.ns[table_index])




