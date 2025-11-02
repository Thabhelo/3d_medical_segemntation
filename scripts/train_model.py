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
from src.data.dataset_configs import get_dataset_config
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
    parser.add_argument("--scheduler", default="none", help="LR scheduler: none | reduce_on_plateau | cosine | onecycle | polynomial | dlrs")
    parser.add_argument("--save_every_epoch", action="store_true", help="Save checkpoint every epoch instead of every 10")
    parser.add_argument("--output_dir", default="results/tmp_run", help="Output directory for checkpoints and logs")
    parser.add_argument("--resume_from", default=None, help="Path to a checkpoint .pth file to resume from")
    parser.add_argument("--patch_size", type=str, default="128,128,128", help="Patch size for training (e.g., '160,160,160')")
    parser.add_argument("--class_weights", type=str, default=None, help="Comma-separated class weights (e.g., '1.0,1.0,3.0')")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max norm")
    args = parser.parse_args()

    # Apply dataset-specific configs if using defaults
    dataset_config = get_dataset_config(args.dataset, args.data_root)
    if args.in_channels == 1 and "in_channels" in dataset_config:
        args.in_channels = dataset_config["in_channels"]
    if args.out_channels == 2 and "out_channels" in dataset_config:
        args.out_channels = dataset_config["out_channels"]
    if args.patch_size == "128,128,128" and "patch_size" in dataset_config:
        args.patch_size = ",".join(map(str, dataset_config["patch_size"]))
    if args.loss == "dice_ce" and "loss" in dataset_config:
        args.loss = dataset_config["loss"]

    # Check if training already completed (unless resuming)
    output_path = Path(args.output_dir)
    if args.resume_from is None and output_path.exists() and (output_path / "best.pth").exists():
        print(f"Training already completed in {output_path}")
        print("Use a different --output_dir to run again")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset root: {args.data_root}")

    patch_size_tuple = tuple(int(x) for x in args.patch_size.split(','))
    if len(patch_size_tuple) != 3:
        raise ValueError(f"patch_size must have 3 dimensions, got {len(patch_size_tuple)}")
    if any(p <= 0 for p in patch_size_tuple):
        raise ValueError(f"patch_size must be positive, got {patch_size_tuple}")

    train_loader, val_loader = create_dataloaders(
        dataset_name=args.dataset,
        root_dir=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=patch_size_tuple,
    )
    print(f"Created dataloaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    print(f"Patch size: {patch_size_tuple}")

    # Ensure UNETR receives the correct img_size matching the training patch size
    model_kwargs = {}
    if args.architecture.lower() == "unetr":
        if any(p % 16 != 0 for p in patch_size_tuple):
            raise ValueError(f"UNETR requires patch_size divisible by 16, got {patch_size_tuple}")
        model_kwargs["img_size"] = patch_size_tuple
    model = create_model(
        architecture=args.architecture,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        **model_kwargs,
    )

    class_weights = None
    if args.class_weights:
        class_weights = [float(x) for x in args.class_weights.split(',')]
        if len(class_weights) != args.out_channels:
            raise ValueError(f"class_weights length {len(class_weights)} must match out_channels {args.out_channels}")
    elif "class_weights" in dataset_config:
        class_weights = dataset_config["class_weights"]

    loss_fn = get_loss(args.loss, class_weights=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.architecture} with {num_params:,} parameters ({num_params/1e6:.2f}M)")
    
    # Create learning rate scheduler
    # For OneCycleLR and DLRS, we need steps_per_epoch
    steps_per_epoch = len(train_loader) if hasattr(train_loader, '__len__') else None
    
    scheduler = None
    if args.scheduler != "none":
        if args.scheduler == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=10
            )
        elif args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.max_epochs, eta_min=args.lr * 0.01
            )
        elif args.scheduler == "onecycle":
            if steps_per_epoch is None:
                raise ValueError("OneCycleLR requires train_loader with known length. Set drop_last=False or ensure dataloader is sized.")
            total_steps = args.max_epochs * steps_per_epoch
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.lr * 10, total_steps=total_steps, epochs=args.max_epochs, steps_per_epoch=steps_per_epoch
            )
        elif args.scheduler == "polynomial":
            scheduler = torch.optim.lr_scheduler.PolynomialLR(
                optimizer, total_iters=args.max_epochs, power=0.9
            )
        elif args.scheduler == "dlrs":
            try:
                from dlrs import DLRSScheduler
                scheduler = DLRSScheduler(optimizer)
            except ImportError:
                raise ImportError(
                    "DLRS scheduler not found. Install with: pip install pytorch-dlrs"
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
        grad_clip=args.grad_clip,
    )

    # Optional resume logic
    start_epoch = 1
    if args.resume_from is not None:
        ckpt_path = Path(args.resume_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=device)
        if "model" not in ckpt:
            raise ValueError(f"Checkpoint missing 'model' key: {ckpt.keys()}")
        try:
            model.load_state_dict(ckpt["model"], strict=True)
        except RuntimeError as e:
            raise RuntimeError(f"Checkpoint incompatible with model architecture: {e}")
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        print(f"Resumed from epoch {ckpt.get('epoch', 0)}, continuing from epoch {start_epoch}")

    metrics = trainer.train(train_loader, val_loader, start_epoch=start_epoch)
    print(metrics)


if __name__ == "__main__":
    main()