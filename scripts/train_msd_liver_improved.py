#!/usr/bin/env python3
"""
Improved MSD Liver training with optimizations for better performance.

Key improvements:
- Foreground-biased sampling (100% positive samples)
- Improved preprocessing (HU clipping + z-score normalization)
- Class-balanced loss (3x weight on tumor class)
- Larger patch size (160³ instead of 128³)
- Better augmentation strategy
"""

import argparse
from pathlib import Path
import os
import sys

# Ensure this script can import the local 'src' package
CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.data.utils import create_dataloaders
from src.models.factory import create_model
from src.models.losses import get_loss
from src.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MSD Liver with improved configuration")
    parser.add_argument("--architecture", default="unet", help="Model architecture: unet | unetr | segresnet")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--max_epochs", type=int, default=50, help="Max training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--scheduler", default="reduce_on_plateau", help="LR scheduler")
    parser.add_argument("--output_dir", default="results/colab_runs/msd_liver_improved", help="Output directory")
    parser.add_argument("--resume_from", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # MSD Liver configuration
    in_channels = 1
    out_channels = 3  # Background, Liver, Tumor
    data_root = "/content/drive/MyDrive/datasets/MSD/Task03_Liver"
    
    # Check if training already completed (unless resuming)
    output_path = Path(args.output_dir)
    if args.resume_from is None and output_path.exists() and (output_path / "best.pth").exists():
        print(f"Training already completed in {output_path}")
        print("Use a different --output_dir to run again")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset root: {data_root}")
    print(f"Architecture: {args.architecture}")
    print(f"Improved configuration: foreground sampling, class-balanced loss, larger patches")

    # Create dataloaders with improved configuration
    train_loader, val_loader = create_dataloaders(
        dataset_name="msd_liver",
        root_dir=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # Larger patch size for better performance (160³ instead of 128³)
        patch_size=(160, 160, 160),
    )
    print(f"Created dataloaders with patch_size=(160, 160, 160)")

    model = create_model(
        architecture=args.architecture,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    
    # Use class-balanced loss for MSD Liver
    loss_fn = get_loss("dice_ce_balanced")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Create learning rate scheduler
    scheduler = None
    if args.scheduler == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=8
        )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        output_dir=Path(args.output_dir),
        max_epochs=args.max_epochs,
        num_classes=out_channels,
        scheduler=scheduler,
        save_every_epoch=True,
    )

    # Optional resume logic
    start_epoch = 1
    if args.resume_from is not None:
        ckpt_path = Path(args.resume_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume_from checkpoint not found: {ckpt_path}")
        print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=device)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        print(f"Resume start_epoch set to {start_epoch}")

    metrics = trainer.train(train_loader, val_loader, start_epoch=start_epoch)
    print(f"Final results: {metrics}")


if __name__ == "__main__":
    main()
