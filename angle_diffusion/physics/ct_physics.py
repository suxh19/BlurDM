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
from importlib import resources

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

        path = Path(focus_table_path)
        if not path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            candidate = project_root / path
            if candidate.exists():
                path = candidate

        if path.is_absolute() and path.exists():
            data = np.load(path)
        else:
            resource_path = resources.files("astra_package.ct.resources").joinpath(
                "focus_tables", path.name
            )
            if not resource_path.is_file():
                raise FileNotFoundError(f"focus table不存在: {path} (资源路径: {resource_path})")
            with resources.as_file(resource_path) as resource_file:
                data = np.load(resource_file)

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


def load_focus_table(focus_table_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load focus table arrays from .npz.

    Returns:
        ns: [100] int array
        mask_table: [100, 512] uint8 array (0/1)
    """
    data = np.load(focus_table_path)
    if "ns" not in data or "mask_table" not in data:
        raise KeyError(f"focus table missing keys: {data.files}")
    ns = data["ns"].astype(int)
    mask_table = data["mask_table"].astype(np.uint8)
    return ns, mask_table


def _parse_t_list(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("empty t_list")
    return out


def _mask_for_t(mask_table: np.ndarray, t: int) -> np.ndarray:
    if t == 0:
        return np.ones((mask_table.shape[1],), dtype=np.uint8)
    if not 1 <= t <= 100:
        raise ValueError(f"t must be in [0,100], got {t}")
    table_index = 100 - t
    return mask_table[table_index]


def _visualize_masks(
    *,
    mask_table: np.ndarray,
    ns: np.ndarray,
    t_list: list[int],
    out_path: str,
) -> None:
    import matplotlib.pyplot as plt

    rows = []
    ylabels = []
    for t in t_list:
        if t == 0:
            m = np.ones((mask_table.shape[1],), dtype=np.uint8)
            ylabels.append("t=0 | full")
        else:
            m = _mask_for_t(mask_table, t)
            table_index = 100 - t
            ylabels.append(f"t={t} | ns={int(ns[table_index])}")
        rows.append(m)

    mat = np.stack(rows, axis=0)
    plt.figure(figsize=(12, max(2.5, 0.6 * len(t_list))))
    plt.imshow(mat, aspect="auto", cmap="gray", vmin=0, vmax=1)
    plt.yticks(ticks=np.arange(len(t_list)), labels=ylabels)
    plt.xlabel("angle index (0..511)")
    plt.title("Focus table masks (selected views)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _load_image_from_dataset(dataset_root: str, split: str, idx: int) -> tuple[np.ndarray, str]:
    # Local import to keep the physics module lightweight when used as a library.
    from angle_diffusion.data.ct_dataset import CTDataset

    ds = CTDataset(dataset_root, split=split, return_path=True)
    gt, _sart, gt_path, _sart_path = ds[idx]  # type: ignore[misc]
    gt_np = gt.squeeze(0).cpu().numpy().astype(np.float32)
    return gt_np, gt_path


def _load_image_from_path(image_path: str) -> np.ndarray:
    from PIL import Image

    img = Image.open(image_path).convert("L")
    arr = np.array(img).astype(np.float32) / 255.0
    return arr


def _visualize(
    *,
    x0: np.ndarray,
    degraded: list[tuple[int, int, np.ndarray]],
    out_path: str,
    vmin: float | None,
    vmax: float | None,
) -> None:
    import matplotlib.pyplot as plt

    # Layout: 2 rows (image + abs error), N columns.
    n = len(degraded)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, (t, ns, x_t) in enumerate(degraded):
        ax_img = axes[0, col]
        ax_err = axes[1, col]

        ax_img.imshow(x_t, cmap="gray", vmin=vmin, vmax=vmax)
        ax_img.set_title(f"t={t} | ns={ns}")
        ax_img.axis("off")

        err = np.abs(x_t - x0)
        ax_err.imshow(err, cmap="hot")
        ax_err.set_title("|x_t - x0|")
        ax_err.axis("off")

    fig.suptitle("CT_Physics Degradation Visualization", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _main() -> None:
    parser = argparse.ArgumentParser(description="CT_Physics self-test + visualization")
    parser.add_argument(
        "--focus_table_path",
        default="../code_diffusion/cold_diffuson/linear_indices_100.focus_table.npz",
        type=str,
    )
    parser.add_argument("--preset", default="standard_512", type=str)
    parser.add_argument("--sart_iterations", default=10, type=int)
    parser.add_argument("--t_list", default="0,25,50,75,100", type=str)
    parser.add_argument("--out_dir", default="./angle_diffusion_runs/ct_physics_test", type=str)
    parser.add_argument("--vmin", default=None, type=float)
    parser.add_argument("--vmax", default=None, type=float)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--mask_only",
        action="store_true",
        help="Only visualize the focus-table masks; skip ASTRA degradation.",
    )

    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--dataset_root", type=str, help="Dataset root containing split/ground_truth, split/sart")
    src.add_argument("--image_path", type=str, help="Path to a grayscale PNG image")

    parser.add_argument("--split", default="val", type=str, choices=["train", "val", "test"])
    parser.add_argument("--idx", default=0, type=int)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t_list = _parse_t_list(args.t_list)

    ns, mask_table = load_focus_table(args.focus_table_path)
    mask_path = os.path.join(args.out_dir, "focus_table_masks.png")
    _visualize_masks(mask_table=mask_table, ns=ns, t_list=t_list, out_path=mask_path)
    print(f"[CT_Physics Test] saved: {mask_path}")

    if args.mask_only:
        return

    if not args.dataset_root and not args.image_path:
        raise ValueError("Provide --dataset_root or --image_path (or use --mask_only).")

    physics = CT_Physics(
        focus_table_path=args.focus_table_path,
        sart_iterations=args.sart_iterations,
        preset=args.preset,
    )

    if args.dataset_root:
        x0, src_path = _load_image_from_dataset(args.dataset_root, args.split, args.idx)
    else:
        x0 = _load_image_from_path(args.image_path)
        src_path = args.image_path

    # Smoke test: shape checks.
    if x0.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {x0.shape}")

    degraded: list[tuple[int, int, np.ndarray]] = []
    for t in t_list:
        ns = physics.t_to_num_sources(t) if t != 0 else -1
        x_t = physics.degrade(x0, t)
        degraded.append((t, ns, x_t))

        np.save(os.path.join(args.out_dir, f"img_t{t:03d}.npy"), x_t.astype(np.float32))

    out_path = os.path.join(args.out_dir, "comparison.png")
    _visualize(x0=x0, degraded=degraded, out_path=out_path, vmin=args.vmin, vmax=args.vmax)

    print(f"[CT_Physics Test] source={src_path}")
    print(f"[CT_Physics Test] saved: {out_path}")
    print(f"[CT_Physics Test] t_list={t_list}")

    if args.show:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        img = mpimg.imread(out_path)
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    _main()
