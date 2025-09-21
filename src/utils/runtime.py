"""
Runtime environment detection utilities.
"""
import os
import platform
from pathlib import Path
from typing import Optional


def is_colab() -> bool:
    """
    Detect if running in Google Colab environment.
    
    Returns
    -------
    bool
        True if running in Colab, False otherwise.
    """
    try:
        import google.colab  # type: ignore
        return True
    except ImportError:
        return False


def is_linux() -> bool:
    """
    Detect if running on Linux (non-Colab).
    
    Returns
    -------
    bool
        True if running on Linux and not in Colab, False otherwise.
    """
    return platform.system() == "Linux" and not is_colab()


def get_dataset_root() -> Path:
    """
    Get the appropriate dataset root directory based on runtime environment.
    
    Returns
    -------
    Path
        Path to the dataset root directory.
    """
    if is_colab():
        # Colab environment - use Google Drive
        return Path("/content/drive/MyDrive/datasets")
    elif is_linux():
        # Linux environment - use Downloads/datasets
        downloads_path = Path.home() / "Downloads" / "datasets"
        return downloads_path
    else:
        # Fallback for other environments (Windows, macOS, etc.)
        # Try Downloads first, then fall back to current directory
        downloads_path = Path.home() / "Downloads" / "datasets"
        if downloads_path.exists():
            return downloads_path
        return Path("datasets")


def get_output_root() -> Path:
    """
    Get the appropriate output root directory based on runtime environment.
    
    Returns
    -------
    Path
        Path to the output root directory.
    """
    if is_colab():
        # Colab environment - use Google Drive
        return Path("/content/drive/MyDrive/3d_medical_segmentation/results")
    elif is_linux():
        # Linux environment - use project results directory
        return Path("results")
    else:
        # Fallback for other environments
        return Path("results")


def get_runtime_info() -> dict:
    """
    Get comprehensive runtime environment information.
    
    Returns
    -------
    dict
        Dictionary containing runtime information.
    """
    return {
        "platform": platform.system(),
        "is_colab": is_colab(),
        "is_linux": is_linux(),
        "dataset_root": str(get_dataset_root()),
        "output_root": str(get_output_root()),
        "home_dir": str(Path.home()),
    }

