from __future__ import annotations

from typing import Dict, Any


def get_dataset_config(dataset_name: str, base_data_root: str) -> Dict[str, Any]:
    """Get configuration for a dataset with sensible defaults."""
    configs = {
        "brats": {
            "in_channels": 4,
            "out_channels": 4,
            "data_root": base_data_root,
        },
        "msd_liver": {
            "in_channels": 1,
            "out_channels": 3,
            "data_root": f"{base_data_root}/MSD/Task03_Liver",
            "patch_size": (160, 160, 160),
            "loss": "dice_ce_balanced",
            "class_weights": [1.0, 1.0, 3.0],
        },
        "totalsegmentator": {
            "in_channels": 1,
            "out_channels": 118,
            "data_root": f"{base_data_root}/TotalSegmentator",
        },
    }
    return configs.get(dataset_name, {})
