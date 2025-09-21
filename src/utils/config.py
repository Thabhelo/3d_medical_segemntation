"""
Configuration utilities with environment-aware path handling.
"""
import yaml
from pathlib import Path
from typing import Dict, Any

from .runtime import get_dataset_root, get_output_root


def load_config_with_env_paths(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file and update paths based on runtime environment.
    
    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
        
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary with environment-appropriate paths.
    """
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Replace template variables with environment-appropriate paths
    content = content.replace('{{ENV_DATASET_ROOT}}', str(get_dataset_root()))
    content = content.replace('{{ENV_OUTPUT_ROOT}}', str(get_output_root()))
    
    config = yaml.safe_load(content)
    return config


def get_config_paths() -> Dict[str, str]:
    """
    Get environment-appropriate configuration paths.
    
    Returns
    -------
    Dict[str, str]
        Dictionary with environment-appropriate paths.
    """
    return {
        'dataset_root': str(get_dataset_root()),
        'output_root': str(get_output_root()),
    }
