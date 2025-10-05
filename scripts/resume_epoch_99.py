#!/usr/bin/env python3
"""
Resume MSD Liver training from epoch 99 to complete epoch 100.
"""

import subprocess
import sys
from pathlib import Path

def main():
    dataset = "msd_liver"
    architecture = "unet"
    
    # MSD Liver configuration
    config = {
        "in_channels": 1,
        "out_channels": 3,  # Background, Liver, Tumor
        "data_root": "/content/drive/MyDrive/datasets/MSD/Task03_Liver"
    }
    
    output_dir = f"results/colab_runs/{dataset}_{architecture}"
    checkpoint_path = f"{output_dir}/epoch_99.pth"
    
    print("RESUMING FROM EPOCH 99 TO COMPLETE EPOCH 100")
    print("=" * 60)
    
    # Check if epoch 99 checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"ERROR: No epoch 99 checkpoint found at {checkpoint_path}")
        return
    
    print(f"Resuming {dataset}_{architecture} from epoch 99...")
    
    # Build command for resuming from epoch 99
    cmd = [
        "python", "-u", "scripts/train_model.py",
        "--dataset", dataset,
        "--architecture", architecture,
        "--in_channels", str(config["in_channels"]),
        "--out_channels", str(config["out_channels"]),
        "--data_root", config["data_root"],
        "--max_epochs", "100",  # Complete to 100
        "--batch_size", "2",
        "--num_workers", "2",
        "--lr", "1e-4",
        "--scheduler", "reduce_on_plateau",
        "--save_every_epoch",
        "--output_dir", output_dir,
        "--resume_from", checkpoint_path
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the training command
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"SUCCESS: {dataset}_{architecture} completed epoch 100!")
        else:
            print(f"FAILED: {dataset}_{architecture} failed with exit code {result.returncode}")
            
    except Exception as e:
        print(f"ERROR: Error running {dataset}_{architecture}: {e}")

if __name__ == "__main__":
    main()
