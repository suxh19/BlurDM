"""
CT Physics - Cold Diffusion 退化操作（批处理 CUDA 版本）

使用线性PSNR增长焦点表 (100个配置, N=119~218)
t=0: Ground Truth (无退化)
t=1: N=218 (最多焦点, 最好质量)
t=100: N=119 (最少焦点, 最差质量)
映射: table_index = 100 - t
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, Dict, Tuple

import astra
import numpy as np

from astra_package.ct.config.config import load_fromnpz
from astra_package.ct.data import N_SEGMENTS
from astra_package.ct.geometry import ProjectionVectorGenerator, get_fov_3d
from astra_package.ct.utils.accelerated import (
    make_gpu_links_batched_3d,
    sino_bchw_to_astra_view,
)


class CT_PhysicsBatch:
    def __init__(
        self,
        focus_table_path: str = "dataset/linear_indices_100.focus_table.npz",
        sensitivity_map_dir: str = "dataset/sensitivity_map",
        sart_iterations: int = 10,
        preset: str = "standard_512",
        fp_algorithm: str = "FP3D_CUDA_BATCH",
        sirt_algorithm: str = "SIRT3D_CUDA_GPU_BATCH",
        batch_size: int | None = None,
        num_workers: int | None = None,
    ) -> None:
        self.focus_table_path = focus_table_path
        self.sensitivity_map_dir = sensitivity_map_dir
        self.sart_iterations = sart_iterations
        self.preset = preset
        self.fp_algorithm = fp_algorithm
        self.sirt_algorithm = sirt_algorithm
        self.batch_size = int(batch_size) if batch_size is not None else None
        self.num_workers = int(num_workers) if num_workers is not None else None

        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.num_workers is not None and self.num_workers <= 0:
            raise ValueError("num_workers must be a positive integer")

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
                raise FileNotFoundError(
                    f"focus table不存在: {path} (资源路径: {resource_path})"
                )
            with resources.as_file(resource_path) as resource_file:
                data = np.load(resource_file)

        self.ns = data["ns"].astype(int)

        self._sensitivity_map_dir_path = self._resolve_path(self.sensitivity_map_dir)
        self._angle_feature_cache: dict[int, np.ndarray] = {}

        print(
            f"[CT_Physics] 初始化完成 | SART={sart_iterations} | 源范围: {self.ns.min()}~{self.ns.max()}"
        )

    def _resolve_path(self, path_like: str) -> Path:
        path = Path(path_like)
        if path.is_absolute():
            return path

        # 优先按仓库根目录解析（.../BlurDM），其次按包根目录解析（.../angle_diffusion）。
        repo_root = Path(__file__).resolve().parents[3]
        candidate = repo_root / path
        if candidate.exists():
            return candidate

        pkg_root = Path(__file__).resolve().parents[2]
        candidate2 = pkg_root / path
        return candidate2 if candidate2.exists() else path

    def t_to_table_index(self, t: int) -> int:
        if not 1 <= t <= 100:
            raise ValueError(f"t 必须在 [1, 100] 范围内，实际: {t}")
        return 100 - int(t)

    def t_to_sensitivity_index(self, t: int) -> int:
        """将退化等级 t 映射到 sensitivity_map 的索引。

        约定:
            - t=1  -> table_index=99 -> 099.npy
            - t=100-> table_index=0  -> 000.npy
            - t=0  (GT) 没有对应文件；上层应特殊处理。
        """
        return self.t_to_table_index(int(t))

    @staticmethod
    def _extract_angle_feature(arr: np.ndarray) -> np.ndarray:
        """将 sensitivity_map 的敏感度图压缩为角度条件特征向量。

        预期输入为 3D 张量 (H,W,C) 或 (C,H,W)（C 为角度分段数）。
        输出为 (C,) float32。
        """
        x = np.asarray(arr)
        if x.ndim != 3:
            raise ValueError(f"sensitivity_map 期望 3D 数组，实际维度: {x.ndim}")

        # 经验规则：最小的维度通常是通道维（角度分段数）。
        ch_dim = int(np.argmin(x.shape))
        x_ch_last = np.moveaxis(x, ch_dim, -1)
        c = int(x_ch_last.shape[-1])
        feat = x_ch_last.reshape(-1, c).mean(axis=0, dtype=np.float32)
        return feat.astype(np.float32, copy=False)

    def get_angle_feature(self, t: int) -> np.ndarray:
        """获取退化等级 t 对应的角度条件特征向量。

        文件命名规则: sensitivity_map/{table_index:03d}.npy
        其中 table_index = 100 - t。
        """
        if not 0 <= t <= 100:
            raise ValueError(f"t 必须在 [0, 100] 范围内，实际: {t}")

        if int(t) == 0:
            cached0 = self._angle_feature_cache.get(-1)
            if cached0 is not None:
                return cached0
            # GT 情况下视野全面覆盖，返回全 1 向量；长度对齐到任意一个有效 t 的特征维度。
            ref = self.get_angle_feature(1)
            ones = np.ones_like(ref)
            self._angle_feature_cache[-1] = ones
            return ones

        table_index = self.t_to_sensitivity_index(int(t))
        cached = self._angle_feature_cache.get(table_index)
        if cached is not None:
            return cached

        base_dir = self._sensitivity_map_dir_path
        fpath = base_dir / f"{table_index:03d}.npy"
        if not fpath.exists():
            raise FileNotFoundError(f"sensitivity_map 不存在: {fpath}")

        arr = np.load(fpath)
        feat = self._extract_angle_feature(arr)
        self._angle_feature_cache[table_index] = feat
        return feat

    def _load_config(self, num_sources: int) -> Dict[str, Any]:
        config = load_fromnpz(self.focus_table_path, num_sources, self.preset)
        config["reconstruction"]["SART"]["sart_iterations"] = self.sart_iterations
        if self.num_workers is not None:
            config["reconstruction"]["SART"]["num_workers"] = self.num_workers
        return config

    def _create_geometries(
        self, config: Dict[str, Any], n_slices: int
    ) -> Tuple[Dict[str, Any], Dict[str, Any], int, int, int, int, int, int]:
        h, w, z, min_x, max_x, min_y, max_y, min_z, max_z = get_fov_3d(
            config, n_slices=n_slices
        )
        h, w, z = int(h), int(w), int(z)

        vol_geom = astra.create_vol_geom(
            h, w, z, min_x, max_x, min_y, max_y, min_z, max_z
        )

        det_rows = int(config["geometry"]["detector"]["detector_row_count"])
        det_cols = int(config["geometry"]["detector"]["num_bins"])
        n_sources = int(
            config["geometry"]["source"].get("num_active_sources", 0)
        ) or len(config["geometry"]["source"]["active_source_indices"])

        vectors = (
            ProjectionVectorGenerator(config)
            .generate_3d()
            .astype(np.float32, copy=False)
        )
        proj_geom = astra.create_proj_geom("cone_vec", det_rows, det_cols, vectors)

        return vol_geom, proj_geom, h, w, z, det_rows, det_cols, n_sources

    def _get_sirt_options(self, config: Dict[str, Any]) -> Dict[str, Any]:
        sart = config.get("reconstruction", {}).get("SART", {})
        options: Dict[str, Any] = {}
        if sart.get("relaxation") is not None:
            options["Relaxation"] = float(sart["relaxation"])
        if sart.get("num_workers") is not None:
            options["NumWorkers"] = int(sart["num_workers"])
        # 注意: SIRT3D_CUDA_GPU_BATCH 不支持 MinConstraint/MaxConstraint
        return options

    def degrade(self, image: np.ndarray, t: int) -> np.ndarray:
        if t == 0:
            return image.copy()

        table_index = self.t_to_table_index(t)
        num_sources = int(self.ns[table_index])
        config = self._load_config(num_sources)

        volume = np.asarray(image, dtype=np.float32)
        if volume.ndim != 3:
            raise ValueError(f"输入必须是3D体积数据，当前维度: {volume.ndim}")

        result = self._degrade_batch_impl(volume[None, ...], config)
        return result[0]

    def degrade_batch(self, images: np.ndarray, t: int) -> np.ndarray:
        if images.ndim == 2:
            return self.degrade(images, t)

        if t == 0:
            return images.copy()

        table_index = self.t_to_table_index(t)
        num_sources = int(self.ns[table_index])
        config = self._load_config(num_sources)

        volumes = np.asarray(images, dtype=np.float32)
        if volumes.ndim != 4:
            raise ValueError(f"输入必须是4D批量体积数据，当前维度: {volumes.ndim}")

        if self.batch_size is None or volumes.shape[0] <= self.batch_size:
            return self._degrade_batch_impl(volumes, config)

        outputs = []
        for start in range(0, volumes.shape[0], self.batch_size):
            chunk = volumes[start : start + self.batch_size]
            outputs.append(self._degrade_batch_impl(chunk, config))
        return np.concatenate(outputs, axis=0)

    def _degrade_batch_impl(
        self, volumes: np.ndarray, config: Dict[str, Any]
    ) -> np.ndarray:
        try:
            import torch
        except Exception as err:
            raise RuntimeError(
                "torch 不可用，无法执行 CUDA batch 退化。请确认已安装 CUDA/CuDNN 并能正确加载。"
            ) from err

        if not astra.use_cuda():
            raise RuntimeError("ASTRA reports CUDA is not available.")

        batch_size, z, h, w = (int(v) for v in volumes.shape)
        vol_geom, proj_geom, _, _, _, det_rows, det_cols, n_sources = (
            self._create_geometries(config, n_slices=z)
        )

        vol_tensor = torch.as_tensor(volumes, dtype=torch.float32)
        if not vol_tensor.is_cuda:
            vol_tensor = vol_tensor.cuda()
        vol_tensor = vol_tensor.contiguous()

        sino_shape = (batch_size, det_rows, n_sources, det_cols * N_SEGMENTS)
        sino = torch.zeros(sino_shape, device=vol_tensor.device, dtype=torch.float32)
        sino_astra = sino_bchw_to_astra_view(sino, det_cols)
        recons = torch.zeros(
            (batch_size, z, h, w), device=vol_tensor.device, dtype=torch.float32
        )

        gt_links = make_gpu_links_batched_3d(vol_tensor, (w, h, z))
        sino_links = make_gpu_links_batched_3d(
            sino_astra, (det_cols, n_sources * N_SEGMENTS, det_rows)
        )
        recon_links = make_gpu_links_batched_3d(recons, (w, h, z))

        gt_ids = [astra.data3d.link("-vol", vol_geom, lnk) for lnk in gt_links]
        sino_ids = [astra.data3d.link("-sino", proj_geom, lnk) for lnk in sino_links]
        recon_ids = [astra.data3d.link("-vol", vol_geom, lnk) for lnk in recon_links]

        fp_id = None
        sirt_id = None
        try:
            fp_cfg: Dict[str, Any] = astra.astra_dict(self.fp_algorithm)
            fp_cfg["VolumeDataIds"] = gt_ids
            fp_cfg["ProjectionDataIds"] = sino_ids
            fp_id = astra.algorithm.create(fp_cfg)
            astra.algorithm.run(fp_id)

            sirt_cfg: Dict[str, Any] = astra.astra_dict(self.sirt_algorithm)
            sirt_cfg["ProjectionDataIds"] = sino_ids
            sirt_cfg["ReconstructionDataIds"] = recon_ids
            options = self._get_sirt_options(config)
            if options:
                sirt_cfg["options"] = options
            sirt_id = astra.algorithm.create(sirt_cfg)
            astra.algorithm.run(sirt_id, self.sart_iterations)
            torch.cuda.synchronize()
        finally:
            if fp_id is not None:
                astra.algorithm.delete(fp_id)
            if sirt_id is not None:
                astra.algorithm.delete(sirt_id)
            astra.data3d.delete(gt_ids + sino_ids + recon_ids)

        return recons.detach().cpu().numpy()

    def t_to_num_sources(self, t: int) -> int:
        if t == 0:
            return -1
        table_index = self.t_to_table_index(t)
        return int(self.ns[table_index])
