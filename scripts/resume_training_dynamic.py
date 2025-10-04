#!/usr/bin/env python3
"""
Resume training from epoch 50 with dynamic learning rate scheduling.

This script will:
1. Resume from the latest checkpoint (epoch 50)
2. Use dynamic learning rate scheduling (ReduceLROnPlateau)
3. Continue training to epoch 100
4. Save checkpoints every epoch
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Configuration for resuming training with dynamic LR
    datasets = ["brats", "msd_liver", "totalsegmentator"]
    architectures = ["unet", "unetr", "segresnet"]
    
    # Dataset-specific configurations
    dataset_configs = {
        "brats": {
            "in_channels": 4,
            "out_channels": 4,
            "data_root": "/content/drive/MyDrive/datasets"
        },
        "msd_liver": {
            "in_channels": 1,
            "out_channels": 2,
            "data_root": "/content/drive/MyDrive/datasets/MSD/Task03_Liver"
        },
        "totalsegmentator": {
            "in_channels": 1,
            "out_channels": 118,
            "data_root": "/content/drive/MyDrive/datasets/TotalSegmentator"
        }
    }
    
    print("RESUMING TRAINING WITH DYNAMIC LEARNING RATE")
    print("=" * 60)
    
    for dataset in datasets:
        config = dataset_configs[dataset]
        print(f"\nDataset: {dataset.upper()}")
        print(f"   Input channels: {config['in_channels']}")
        print(f"   Output channels: {config['out_channels']}")
        print(f"   Data root: {config['data_root']}")
        
        for architecture in architectures:
            output_dir = f"results/colab_runs/{dataset}_{architecture}"
            checkpoint_path = f"{output_dir}/epoch_50.pth"
            
            # Check if epoch 50 checkpoint exists
            if not Path(checkpoint_path).exists():
                print(f"   WARNING: No epoch 50 checkpoint found for {dataset}_{architecture}")
                continue
                
            print(f"\n   Resuming {dataset}_{architecture} from epoch 50...")
            
            # Build command for resuming with dynamic LR
            cmd = [
                "python", "-u", "scripts/train_model.py",
                "--dataset", dataset,
                "--architecture", architecture,
                "--in_channels", str(config["in_channels"]),
                "--out_channels", str(config["out_channels"]),
                "--data_root", config["data_root"],
                "--max_epochs", "100",  # Continue to 100
                "--batch_size", "2",
                "--num_workers", "2",
                "--lr", "1e-4",
                "--scheduler", "reduce_on_plateau",  # Dynamic LR
                "--save_every_epoch",  # Save every epoch
                "--output_dir", output_dir,
                "--resume_from", checkpoint_path
            ]
            
            print(f"   Command: {' '.join(cmd)}")
            
            try:
                # Run the training command
                result = subprocess.run(cmd, capture_output=False, text=True)
                
                if result.returncode == 0:
                    print(f"   SUCCESS: {dataset}_{architecture} completed successfully")
                else:
                    print(f"   FAILED: {dataset}_{architecture} failed with exit code {result.returncode}")
                    
            except Exception as e:
                print(f"   ERROR: Error running {dataset}_{architecture}: {e}")
    
    print("\nResume training with dynamic LR completed!")
    print("All models should now have:")
    print("- Dynamic learning rate scheduling (ReduceLROnPlateau)")
    print("- Training from epoch 50 to 100")
    print("- Checkpoints saved every epoch")

if __name__ == "__main__":
    main()
