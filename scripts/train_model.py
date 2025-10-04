#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

# Ensure this script can import the local 'src' package even when invoked from subprocesses
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
    """
    Entry point for launching a training run from the command line.

    This script intentionally keeps sensible defaults so it can run out-of-the-box
    on Google Colab or a local machine:
    - Automatically detects dataset root (Colab Drive vs. $HOME/Downloads/datasets)
    - Uses small batch size and epochs by default for quick smoke tests
    - Creates output directory and skips re-training if a best checkpoint exists
    """
    parser = argparse.ArgumentParser(description="Train a 3D segmentation model")
    parser.add_argument("--dataset", default="brats", help="Dataset name: brats | msd_liver | totalsegmentator")
    # Auto-detect dataset root based on environment (Colab vs local)
    default_data_root = (
        "/content/drive/MyDrive/datasets" if os.path.isdir("/content") else str(Path.home() / "Downloads" / "datasets")
    )
    parser.add_argument("--data_root", default=default_data_root, help="Path to datasets root directory")
    parser.add_argument("--architecture", default="unet", help="Model architecture: unet | unetr | segresnet")
    parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels")
    parser.add_argument("--out_channels", type=int, default=2, help="Number of output classes/channels")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--max_epochs", type=int, default=2, help="Max training epochs")
    parser.add_argument("--loss", default="dice_ce", help="Loss function key")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--scheduler", default="none", help="LR scheduler: none | reduce_on_plateau | cosine | onecycle | polynomial")
    parser.add_argument("--save_every_epoch", action="store_true", help="Save checkpoint every epoch instead of every 10")
    parser.add_argument("--output_dir", default="results/tmp_run", help="Output directory for checkpoints and logs")
    parser.add_argument("--resume_from", default=None, help="Path to a checkpoint .pth file to resume from")
    args = parser.parse_args()

    # Check if training already completed (unless resuming)
    output_path = Path(args.output_dir)
    if args.resume_from is None and output_path.exists() and (output_path / "best.pth").exists():
        print(f"Training already completed in {output_path}")
        print("Use a different --output_dir to run again")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset root: {args.data_root}")

    train_loader, val_loader = create_dataloaders(
        dataset_name=args.dataset,
        root_dir=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # Force patch-based training/validation for resource-constrained environments
        patch_size=(128, 128, 128),
    )
    print(f"Created dataloaders with patch_size=(128, 128, 128)")

    model = create_model(
        architecture=args.architecture,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
    )
    loss_fn = get_loss(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Create learning rate scheduler
    scheduler = None
    if args.scheduler != "none":
        if args.scheduler == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=10, verbose=True
            )
        elif args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.max_epochs, eta_min=args.lr * 0.01
            )
        elif args.scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.lr * 10, total_steps=args.max_epochs
            )
        elif args.scheduler == "polynomial":
            scheduler = torch.optim.lr_scheduler.PolynomialLR(
                optimizer, total_iters=args.max_epochs, power=0.9
            )
        else:
            raise ValueError(f"Unknown scheduler: {args.scheduler}")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        output_dir=Path(args.output_dir),
        max_epochs=args.max_epochs,
        num_classes=args.out_channels,
        scheduler=scheduler,
        save_every_epoch=args.save_every_epoch,
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
    print(metrics)


if __name__ == "__main__":
    main()