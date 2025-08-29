from __future__ import annotations

import os
from pathlib import Path


def in_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False


def get_drive_root() -> Path:
    if in_colab():
        return Path("/content/drive/MyDrive")
    # Desktop Google Drive default
    return Path.home() / "Google Drive" / "My Drive"


def get_project_root(default_name: str = "3d_medical_segmentation") -> Path:
    root = get_drive_root() / default_name
    return root


def get_datasets_root() -> Path:
    return get_drive_root() / "datasets"


