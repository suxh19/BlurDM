cfg = {
    "data": {
        "root": "../code_diffusion/dataset_che",
        "train_split": "train",
        "val_split": "val",
        "max_samples": None,
    },
    "training": {
        "batch_size": 4,
        "lr": 2e-4,
        "epochs": 200,
        "num_workers": 4,
        "save_interval": 10,
        "val_interval": 5,
        "seed": 2023,
    },
    "physics": {
        "focus_table_path": "../code_diffusion/cold_diffuson/linear_indices_100.focus_table.npz",
        "max_t": 100,
        "t_min": 1,
        "t_max": 100,
        "sart_iterations": 10,
        "config_preset": "standard_512",
    },
    "model": {
        "image_size": 512,
        "base_channels": 64,
        "channel_mults": [1, 2, 4, 8],
        "num_res_blocks": 2,
    },
    "output": {"out_dir": "./angle_diffusion_runs/stage1"},
}

