#!/usr/bin/env python3
"""
Diagnostic script to check data normalization across all datasets.
Verifies that input images are properly normalized to [0, 1] range.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Ensure we can import from src
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from src.data.utils import create_dataloaders


def check_dataset_normalization(dataset_name: str, data_root: str, num_samples: int = 10):
    """
    Check min/max values of input images from a dataset.

    Args:
        dataset_name: Name of the dataset (brats, msd_liver, totalsegmentator)
        data_root: Root directory containing datasets
        num_samples: Number of samples to check
    """
    print(f"\n{'='*70}")
    print(f"Checking normalization for: {dataset_name.upper()}")
    print(f"{'='*70}")

    try:
        # Determine expected channels based on dataset
        if dataset_name == "brats":
            in_channels = 4  # T1, T1ce, T2, FLAIR
            out_channels = 4  # Background + 3 tumor regions
        elif dataset_name == "msd_liver":
            in_channels = 1  # CT
            out_channels = 3  # Background + liver + tumor
        elif dataset_name == "totalsegmentator":
            in_channels = 1  # CT
            out_channels = 118  # 117 organs + background
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            dataset_name=dataset_name,
            root_dir=data_root,
            batch_size=1,
            num_workers=0,
            patch_size=(128, 128, 128),
        )

        print(f"Expected input channels: {in_channels}")
        print(f"Expected output channels: {out_channels}")
        print(f"\nChecking {num_samples} samples from training set...")

        all_mins = []
        all_maxs = []
        all_means = []
        all_stds = []

        # Check training samples
        for idx, batch in enumerate(train_loader):
            if idx >= num_samples:
                break

            images = batch["image"]  # Shape: (B, C, D, H, W)
            labels = batch["label"]  # Shape: (B, C, D, H, W) or (B, 1, D, H, W)

            # Convert to numpy for analysis
            img_np = images.numpy()
            lbl_np = labels.numpy()

            img_min = img_np.min()
            img_max = img_np.max()
            img_mean = img_np.mean()
            img_std = img_np.std()

            all_mins.append(img_min)
            all_maxs.append(img_max)
            all_means.append(img_mean)
            all_stds.append(img_std)

            print(f"  Sample {idx + 1}:")
            print(f"    Image shape: {tuple(images.shape)}")
            print(f"    Label shape: {tuple(labels.shape)}")
            print(f"    Image min: {img_min:.6f}, max: {img_max:.6f}")
            print(f"    Image mean: {img_mean:.6f}, std: {img_std:.6f}")
            print(f"    Label unique values: {np.unique(lbl_np)[:10]}")  # First 10 unique values

            # Per-channel statistics
            if images.shape[1] > 1:  # Multi-channel
                print(f"    Per-channel stats:")
                for c in range(images.shape[1]):
                    ch_min = img_np[:, c].min()
                    ch_max = img_np[:, c].max()
                    ch_mean = img_np[:, c].mean()
                    print(f"      Channel {c}: min={ch_min:.6f}, max={ch_max:.6f}, mean={ch_mean:.6f}")

        print(f"\n{'='*70}")
        print(f"SUMMARY FOR {dataset_name.upper()}:")
        print(f"{'='*70}")
        print(f"Overall min: {min(all_mins):.6f}")
        print(f"Overall max: {max(all_maxs):.6f}")
        print(f"Average mean: {np.mean(all_means):.6f}")
        print(f"Average std: {np.mean(all_stds):.6f}")

        # Check if data is properly normalized
        if min(all_mins) < -0.1 or max(all_maxs) > 1.1:
            print(f"\n⚠️  WARNING: Data appears to be OUTSIDE [0, 1] range!")
            print(f"   Expected range: [0, 1]")
            print(f"   Actual range: [{min(all_mins):.6f}, {max(all_maxs):.6f}]")
        elif min(all_mins) >= 0 and max(all_maxs) <= 1:
            print(f"\n✓ Data appears to be properly normalized to [0, 1] range")
        else:
            print(f"\n⚠️  WARNING: Data normalization is unclear!")

        return {
            "min": min(all_mins),
            "max": max(all_maxs),
            "mean": np.mean(all_means),
            "std": np.mean(all_stds),
        }

    except Exception as e:
        print(f"\n❌ ERROR checking {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point."""
    from src.data.utils import get_default_dataset_root
    
    # Auto-detect data root (Colab, Google Drive Desktop, or local)
    data_root = get_default_dataset_root()

    print(f"Using data root: {data_root}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    datasets = ["brats", "msd_liver", "totalsegmentator"]
    results = {}

    for dataset_name in datasets:
        result = check_dataset_normalization(
            dataset_name=dataset_name,
            data_root=data_root,
            num_samples=5,  # Check 5 samples per dataset
        )
        results[dataset_name] = result

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY - DATA NORMALIZATION CHECK")
    print(f"{'='*70}")
    for dataset_name, stats in results.items():
        if stats:
            print(f"\n{dataset_name.upper()}:")
            print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
            print(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
            if stats['min'] < -0.1 or stats['max'] > 1.1:
                print(f"  Status: ❌ NOT NORMALIZED PROPERLY")
            else:
                print(f"  Status: ✓ Appears normalized")
        else:
            print(f"\n{dataset_name.upper()}: ❌ Check failed")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
