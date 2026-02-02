"""检查点保存和加载功能"""

from __future__ import annotations

import os

import torch


def save_checkpoint(
    out_dir: str,
    name: str,
    be: torch.nn.Module,
    dm: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float | None = None,
) -> str:
    """保存训练检查点
    
    Args:
        out_dir: 输出目录
        name: 检查点文件名
        be: AngleEncoder 模型
        dm: AngleDM 模型
        optimizer: 优化器
        epoch: 当前轮次
        best_val: 最佳验证指标
        
    Returns:
        检查点文件路径
    """
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
