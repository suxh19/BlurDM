"""
CT Physics 退化算子测试与可视化脚本
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from angle_diffusion.physics.ct_physics import CT_Physics


def _normalize_for_display(img: np.ndarray) -> np.ndarray:
    """归一化到 [0, 1] 用于显示"""
    img_min = float(img.min())
    img_max = float(img.max())
    return (img - img_min) / (img_max - img_min + 1e-8)


def _calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算 PSNR"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return float(10 * np.log10(1.0 / mse))


def _calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算 SSIM"""
    img1_norm = _normalize_for_display(img1)
    img2_norm = _normalize_for_display(img2)
    ssim_value = ssim(img1_norm, img2_norm, data_range=1.0)
    # ssim 可能返回 tuple (ssim_value, gradient_map) 或单个值
    if isinstance(ssim_value, tuple):
        return float(ssim_value[0])
    return float(ssim_value)


def load_gt_images(data_root: str, split: str = "test", max_samples: int = 2) -> np.ndarray:
    """
    从数据集加载 ground truth 图像
    
    Args:
        data_root: 数据集根目录
        split: 数据集划分 (train/val/test)
        max_samples: 最大样本数
        
    Returns:
        图像数组 (N, H, W)
    """
    gt_dir = Path(data_root) / split / "ground_truth"
    if not gt_dir.exists():
        raise ValueError(f"Ground truth 目录不存在: {gt_dir}")
    
    gt_paths = sorted(gt_dir.glob("*.png"))[:max_samples]
    if len(gt_paths) == 0:
        raise ValueError(f"在 {gt_dir} 中没有找到 PNG 图像")
    
    images = []
    for path in gt_paths:
        img = Image.open(path).convert("L")
        img_np = np.array(img).astype(np.float32) / 255.0
        images.append(img_np)
    
    return np.stack(images, axis=0)


def generate_comparison_images(
    x0_np: np.ndarray,
    physics: CT_Physics,
    out_dir: str,
    sample_idx: int = 0,
    t_show: list[int] | None = None,
) -> list[str]:
    """
    为每个退化等级生成单独的图像

    Args:
        x0_np: 原始图像数组 (N, H, W)
        physics: CT 物理层
        out_dir: 输出目录
        sample_idx: 要可视化的样本索引
        t_show: 要显示的退化等级列表
        
    Returns:
        生成的文件路径列表
    """
    if t_show is None:
        t_show = [1, 10, 25, 50, 75, 100]

    x0_single = x0_np[sample_idx]
    generated_files = []

    for t_val in t_show:
        # 退化图像
        x_t_np = physics.degrade_batch(x0_np[sample_idx : sample_idx + 1], t_val)[0]

        # 计算指标
        psnr_val = _calculate_psnr(x0_single, x_t_np)
        ssim_val = _calculate_ssim(x0_single, x_t_np)

        # 创建单张图像
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        ax.imshow(_normalize_for_display(x_t_np), cmap="gray")
        ax.set_title(
            f"退化图像 (t={t_val})\nPSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}",
            fontsize=14,
            pad=10,
        )
        ax.axis("off")

        plt.tight_layout()

        output_path = os.path.join(out_dir, f"degraded_sample{sample_idx}_t{t_val:03d}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        generated_files.append(output_path)
        print(f"[CT_Physics Test] 生成图像: {output_path}")

    return generated_files


def test_degradation_operator(
    physics: CT_Physics,
    x0_np: np.ndarray,
    out_dir: str,
    t_step: int = 10,
):
    """
    退化算子完整测试与可视化

    Args:
        physics: CT 物理层
        x0_np: 原始图像数组 (N, H, W)
        out_dir: 输出目录
        t_step: t 的步长，用于加速测试
    """
    os.makedirs(out_dir, exist_ok=True)

    # 生成对比图
    print("[CT_Physics Test] 生成对比图...")
    generate_comparison_images(x0_np, physics, out_dir, sample_idx=0)

    # PSNR 曲线测试
    if t_step < 1:
        raise ValueError("t_step 必须 >= 1")

    t_values = list(range(0, 101, t_step))
    if t_values[-1] != 100:
        t_values.append(100)
    
    psnr_map = np.zeros((x0_np.shape[0], len(t_values)), dtype=np.float32)

    print(f"[CT_Physics Test] 开始退化测试: samples={x0_np.shape[0]}, steps={len(t_values)}")
    for idx, t_val in enumerate(t_values):
        if idx % max(1, len(t_values) // 10) == 0:
            print(f"[CT_Physics Test] t={t_val} ({idx + 1}/{len(t_values)})")
        x_t_np = physics.degrade_batch(x0_np, t_val)
        mse = np.mean((x_t_np - x0_np) ** 2, axis=(1, 2))
        psnr = np.full_like(mse, 100.0, dtype=np.float32)
        valid = mse > 0
        psnr[valid] = 10 * np.log10(1.0 / mse[valid])
        psnr_map[:, idx] = psnr

    # 保存 PSNR 数据
    npy_path = os.path.join(out_dir, "degradation_psnr.npy")
    np.save(npy_path, psnr_map)

    # 1) 可视化退化等级 (第一个样本)
    t_show = [0, 1, 10, 25, 50, 75, 100]
    fig, axes = plt.subplots(1, len(t_show), figsize=(3 * len(t_show), 3))
    for i, t_val in enumerate(t_show):
        x_t_np = physics.degrade_batch(x0_np[:1], t_val)[0]
        axes[i].imshow(_normalize_for_display(x_t_np), cmap="gray")
        axes[i].set_title(f"t={t_val}")
        axes[i].axis("off")
    plt.tight_layout()
    levels_path = os.path.join(out_dir, "degradation_levels_sample0.png")
    plt.savefig(levels_path, dpi=150)
    plt.close()

    # 2) PSNR 曲线
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for i in range(psnr_map.shape[0]):
        ax.plot(t_values, psnr_map[i], label=f"sample_{i}")
    ax.set_xlabel("t")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("CT Physics Degradation: PSNR vs t")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    psnr_path = os.path.join(out_dir, "degradation_psnr.png")
    plt.savefig(psnr_path, dpi=150)
    plt.close()

    # 3) 灰度分布 (第一个样本)
    x_t_show = [physics.degrade_batch(x0_np[:1], t_val)[0] for t_val in t_show]
    values_min = float(min(x_t.min() for x_t in x_t_show))
    values_max = float(max(x_t.max() for x_t in x_t_show))
    bins = 64

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for t_val, x_t_np in zip(t_show, x_t_show, strict=True):
        hist, bin_edges = np.histogram(
            x_t_np.flatten(),
            bins=bins,
            range=(values_min, values_max),
            density=True,
        )
        centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
        ax.plot(centers, hist, label=f"t={t_val}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("CT Physics Degradation Histogram (sample 0)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    hist_path = os.path.join(out_dir, "degradation_histogram_sample0.png")
    plt.savefig(hist_path, dpi=150)
    plt.close()

    # 4) 原图 vs 退化图对比
    fig, axes = plt.subplots(2, len(t_show), figsize=(3 * len(t_show), 6))
    for i, t_val in enumerate(t_show):
        x_t_np = physics.degrade_batch(x0_np[:1], t_val)[0]
        psnr_val = _calculate_psnr(x0_np[0], x_t_np)
        ssim_val = _calculate_ssim(x0_np[0], x_t_np)
        
        # 原图
        axes[0, i].imshow(_normalize_for_display(x0_np[0]), cmap="gray")
        axes[0, i].set_title(f"GT (t=0)" if i == 0 else "")
        axes[0, i].axis("off")
        
        # 退化图
        axes[1, i].imshow(_normalize_for_display(x_t_np), cmap="gray")
        axes[1, i].set_title(f"t={t_val}\nPSNR={psnr_val:.1f}")
        axes[1, i].axis("off")
    
    plt.tight_layout()
    comparison_path = os.path.join(out_dir, "gt_vs_degraded_comparison.png")
    plt.savefig(comparison_path, dpi=150)
    plt.close()

    # 5) t 与 源数量 (num_sources) 的关系
    t_range = list(range(1, 101))
    num_sources_list = [physics.t_to_num_sources(t) for t in t_range]
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(t_range, num_sources_list, 'b-', linewidth=2)
    ax.set_xlabel("t (退化程度)")
    ax.set_ylabel("num_sources (CT 源数量)")
    ax.set_title("t 与 CT 源数量的关系")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    sources_path = os.path.join(out_dir, "t_vs_num_sources.png")
    plt.savefig(sources_path, dpi=150)
    plt.close()

    print(f"[CT_Physics Test] 输出文件:")
    print(f"  - {levels_path}")
    print(f"  - {psnr_path}")
    print(f"  - {npy_path}")
    print(f"  - {hist_path}")
    print(f"  - {comparison_path}")
    print(f"  - {sources_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CT Physics 退化算子测试与可视化")
    parser.add_argument(
        "--data-root",
        type=str,
        default="dataset",
        help="数据集根目录",
    )
    parser.add_argument(
        "--focus-table",
        type=str,
        default="dataset/linear_indices_100.focus_table.npz",
        help="Focus table 路径",
    )
    parser.add_argument(
        "--sart-iterations",
        type=int,
        default=10,
        help="SART 迭代次数",
    )
    parser.add_argument(
        "--config-preset",
        type=str,
        default="standard_512",
        help="配置预设",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=2,
        help="测试样本数",
    )
    parser.add_argument(
        "--test-out-dir",
        type=str,
        default="angle_diffusion/output/ct_physics_tests",
        help="输出目录",
    )
    parser.add_argument(
        "--test-t-step",
        type=int,
        default=10,
        help="t 的步长 (>=1)，越小越精细但越慢",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="数据集划分 (train/val/test)",
    )
    args = parser.parse_args()

    # 初始化 CT_Physics
    physics = CT_Physics(
        focus_table_path=args.focus_table,
        sart_iterations=args.sart_iterations,
        preset=args.config_preset,
    )

    # 加载测试图像
    x0_np = load_gt_images(
        data_root=args.data_root,
        split=args.split,
        max_samples=args.test_samples,
    )
    print(f"[CT_Physics Test] 加载了 {x0_np.shape[0]} 个样本, 形状: {x0_np.shape}")

    # 运行测试
    test_degradation_operator(
        physics=physics,
        x0_np=x0_np,
        out_dir=args.test_out_dir,
        t_step=args.test_t_step,
    )


if __name__ == "__main__":
    main()
