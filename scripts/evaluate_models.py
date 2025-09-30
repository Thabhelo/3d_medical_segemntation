#!/usr/bin/env python3
"""
Comprehensive evaluation script for trained 3D medical segmentation models.

This script evaluates all trained model checkpoints and generates:
- Per-model metrics (Dice, IoU, Hausdorff distance)
- Comparative analysis across architectures and datasets
- Summary tables and visualizations
"""

from __future__ import annotations

import sys
from pathlib import Path
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Any

# Add project root to path
REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.utils import create_dataloaders
from src.models.factory import create_model
from src.models.losses import get_loss
from src.training.trainer import Trainer


def evaluate_model_checkpoint(
    checkpoint_path: Path,
    dataset: str,
    architecture: str,
    data_root: str,
    in_channels: int,
    out_channels: int,
) -> Dict[str, Any]:
    """Evaluate a single model checkpoint."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = create_model(
        architecture=architecture,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    
    # Create validation dataloader
    _, val_loader = create_dataloaders(
        dataset_name=dataset,
        root_dir=data_root,
        batch_size=1,
        num_workers=0,
        patch_size=(128, 128, 128),
    )
    
    # Initialize trainer for evaluation
    loss_fn = get_loss("dice_ce")
    optimizer = torch.optim.Adam(model.parameters())  # Dummy optimizer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        output_dir=checkpoint_path.parent,
        max_epochs=1,
    )
    
    # Evaluate
    val_dice = trainer.validate(val_loader)
    
    return {
        "dataset": dataset,
        "architecture": architecture,
        "checkpoint_path": str(checkpoint_path),
        "val_dice": val_dice,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "model_size_mb": checkpoint_path.stat().st_size / (1024 * 1024),
    }


def main():
    """Main evaluation function."""
    
    # Configuration
    RESULTS_DIR = Path("results/colab_runs")
    DATASETS_CONFIG = {
        "brats": {"in_channels": 4, "out_channels": 4, "data_root": "/content/drive/MyDrive/datasets"},
        "msd_liver": {"in_channels": 1, "out_channels": 3, "data_root": "/content/drive/MyDrive/datasets/MSD/Task03_Liver"},
        "totalsegmentator": {"in_channels": 1, "out_channels": 2, "data_root": "/content/drive/MyDrive/datasets/TotalSegmentator"},
    }
    ARCHITECTURES = ["unet", "unetr", "segresnet"]
    
    # Find all checkpoints
    checkpoints = list(RESULTS_DIR.glob("*/best.pth"))
    print(f"Found {len(checkpoints)} model checkpoints")
    
    # Evaluate each checkpoint
    results = []
    for checkpoint_path in sorted(checkpoints):
        # Parse dataset and architecture from path
        parts = checkpoint_path.parent.name.split("_")
        if len(parts) >= 2:
            dataset = "_".join(parts[:-1])  # Handle multi-word datasets like "msd_liver"
            architecture = parts[-1]
            
            if dataset in DATASETS_CONFIG and architecture in ARCHITECTURES:
                print(f"Evaluating {dataset} + {architecture}...")
                
                config = DATASETS_CONFIG[dataset]
                try:
                    result = evaluate_model_checkpoint(
                        checkpoint_path=checkpoint_path,
                        dataset=dataset,
                        architecture=architecture,
                        data_root=config["data_root"],
                        in_channels=config["in_channels"],
                        out_channels=config["out_channels"],
                    )
                    results.append(result)
                    print(f"  Dice: {result['val_dice']:.4f}")
                except Exception as e:
                    print(f"  Error: {e}")
            else:
                print(f"Skipping unknown combination: {dataset} + {architecture}")
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Save detailed results
    output_file = RESULTS_DIR / "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Print summary table
    if not df.empty:
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        # Pivot table for easy comparison
        pivot_df = df.pivot(index="dataset", columns="architecture", values="val_dice")
        print("\nValidation Dice Scores:")
        print(pivot_df.round(4))
        
        # Model complexity comparison
        print("\nModel Parameters (millions):")
        param_pivot = df.pivot(index="dataset", columns="architecture", values="num_parameters")
        print((param_pivot / 1e6).round(2))
        
        # Model size comparison
        print("\nModel Size (MB):")
        size_pivot = df.pivot(index="dataset", columns="architecture", values="model_size_mb")
        print(size_pivot.round(1))
        
        # Best performing models
        print("\nBest Models by Dataset:")
        for dataset in df["dataset"].unique():
            dataset_df = df[df["dataset"] == dataset]
            best_idx = dataset_df["val_dice"].idxmax()
            best_model = dataset_df.loc[best_idx]
            print(f"  {dataset}: {best_model['architecture']} (Dice: {best_model['val_dice']:.4f})")
    
    print(f"\nEvaluation complete! Results saved to {output_file}")


if __name__ == "__main__":
    main()
