"""工具函数"""

from __future__ import annotations

import numpy as np
import torch


def shift_to_unit(tensor: torch.Tensor) -> torch.Tensor:
    """将张量从 [-0.5, 0.5] 映射到 [0, 1]
    
    Args:
        tensor: 输入张量
        
    Returns:
        映射后的张量
    """
    return (tensor + 0.5).clamp(0.0, 1.0)


def shift_to_unit_np(array: np.ndarray) -> np.ndarray:
    """将 numpy 数组从 [-0.5, 0.5] 映射到 [0, 1]
    
    Args:
        array: 输入数组
        
    Returns:
        映射后的数组
    """
    return np.clip(array + 0.5, 0.0, 1.0)
