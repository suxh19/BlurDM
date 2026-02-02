"""数据退化和缓存相关功能"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

from angle_diffusion.config import Stage2Config
from angle_diffusion.physics.ct_physics_batch import CT_PhysicsBatch


def sample_degrade_level(cfg: Stage2Config) -> int:
    """从配置中随机采样一个退化级别
    
    Args:
        cfg: Stage2 配置
        
    Returns:
        退化级别 t
        
    Raises:
        ValueError: 如果 degrade_t_values 为 None
    """
    if cfg.degrade_t_values is None:
        raise ValueError("cfg.degrade_t_values must be provided and cannot be None")
    return int(random.choice(cfg.degrade_t_values))


def get_cache_path(cache_dir: str, gt_path: str, t_level: int) -> str:
    """生成缓存文件路径
    
    Args:
        cache_dir: 缓存目录
        gt_path: 原始图像路径
        t_level: 退化级别
        
    Returns:
        缓存文件路径
    """
    stem = Path(gt_path).stem
    return os.path.join(cache_dir, f"{stem}_t{t_level:03d}.npy")


def degrade_batch_with_cache(
    physics: CT_PhysicsBatch,
    gt_batch_01: np.ndarray,
    gt_paths: list[str],
    t_level: int,
    cache_dir: str | None,
    clip_01: bool = True,
) -> np.ndarray:
    """批量退化处理，支持缓存
    
    Args:
        physics: CT 物理操作器
        gt_batch_01: Ground truth batch，范围 [0, 1]
        gt_paths: Ground truth 文件路径列表
        t_level: 退化级别
        cache_dir: 缓存目录，如果为 None 则不使用缓存
        clip_01: 是否将结果裁剪到 [0, 1]
        
    Returns:
        退化后的数据
    """
    if cache_dir is None:
        out = physics.degrade_batch(gt_batch_01, t_level).astype(np.float32)
        return np.clip(out, 0.0, 1.0) if clip_01 else out

    os.makedirs(cache_dir, exist_ok=True)
    degraded = np.empty_like(gt_batch_01, dtype=np.float32)

    missing_idx: list[int] = []
    missing_inputs: list[np.ndarray] = []
    for i, gt_path in enumerate(gt_paths):
        cpath = get_cache_path(cache_dir, gt_path, t_level)
        if os.path.exists(cpath):
            degraded[i] = np.load(cpath).astype(np.float32)
        else:
            missing_idx.append(i)
            missing_inputs.append(gt_batch_01[i])

    if missing_idx:
        batch_missing = np.stack(missing_inputs, axis=0)
        degraded_missing = physics.degrade_batch(batch_missing, t_level).astype(
            np.float32
        )
        if clip_01:
            degraded_missing = np.clip(degraded_missing, 0.0, 1.0)
        for j, i in enumerate(missing_idx):
            degraded[i] = degraded_missing[j]
            np.save(get_cache_path(cache_dir, gt_paths[i], t_level), degraded_missing[j])

    return degraded
