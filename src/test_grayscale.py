#!/usr/bin/env python
"""
测试灰度图（1通道）输入是否能正常工作
"""
import sys
sys.path.insert(0, '/home/suxh/code/pycode/diffusion/BlurDM/src')

import torch

print("=" * 60)
print("灰度图适配验证测试")
print("=" * 60)

# 测试 1: LatentAngleDiffusion
print("\n[1] 测试 LatentAngleDiffusion (MIMO_UNet 版本)...")
from MIMO_UNet.models.LatentAngleDM import LatentAngleDiffusion

dm = LatentAngleDiffusion(
    total_timestamps=5,
    in_channels=1,
    phi_max=180.0,
    phi_min=60.0,
)
print(f"    - 默认 in_channels: {dm.condition_encoder.pixel_unshuffle.downscale_factor}")
x = torch.randn(2, 1, 64, 64)  # 灰度图 [B, 1, H, W]
out = dm(x)
print(f"    - 输入形状: {x.shape}")
print(f"    - 输出形状: {out.shape}")
print("    ✅ LatentAngleDiffusion 通过！")

# 测试 2: MIMOUNetPlusPrior
print("\n[2] 测试 MIMOUNetPlusPrior...")
from MIMO_UNet.models.MIMOUNetBlurDM import MIMOUNetPlusPrior

net = MIMOUNetPlusPrior(num_res=8, in_channels=1)
print(f"    - in_channels: {net.in_channels}")
prior = torch.randn(2, 256)
x = torch.randn(2, 1, 64, 64)  # 灰度图
outputs = net(x, prior)
print(f"    - 输入形状: {x.shape}")
print(f"    - 输出形状: {[o.shape for o in outputs]}")
print("    ✅ MIMOUNetPlusPrior 通过！")

# 测试 3: VGGPerceptualLoss 灰度图处理
print("\n[3] 测试 VGGPerceptualLoss 灰度图处理...")
from MIMO_UNet.models.losses import VGGPerceptualLoss

loss_fn = VGGPerceptualLoss()
x1 = torch.randn(2, 1, 64, 64)  # 灰度图
x2 = torch.randn(2, 1, 64, 64)
loss = loss_fn(x1, x2)
print(f"    - 灰度图输入形状: {x1.shape}")
print(f"    - 损失值: {loss.item():.4f}")
print("    ✅ VGGPerceptualLoss 灰度图处理通过！")

print("\n" + "=" * 60)
print("🎉 所有测试通过！灰度图适配成功！")
print("=" * 60)
