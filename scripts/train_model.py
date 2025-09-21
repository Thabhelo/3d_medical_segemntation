#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data.utils import create_dataloaders
from src.models.factory import create_model
from src.models.losses import get_loss
from src.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="brats")
    parser.add_argument("--data_root", default="/home/aai-intern/Downloads/datasets")
    parser.add_argument("--architecture", default="unet")
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--loss", default="dice_ce")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", default="results/tmp_run")
    args = parser.parse_args()

    # Check if training already completed
    output_path = Path(args.output_dir)
    if output_path.exists() and (output_path / "best.pth").exists():
        print(f"Training already completed in {output_path}")
        print("Use a different --output_dir to run again")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = create_dataloaders(
        dataset_name=args.dataset,
        root_dir=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=(128, 128, 128),  # Force patch-based training/validation
    )
    print(f"Created dataloaders with patch_size=(128, 128, 128)")

    model = create_model(
        architecture=args.architecture,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
    )
    loss_fn = get_loss(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        output_dir=Path(args.output_dir),
        max_epochs=args.max_epochs,
    )
    metrics = trainer.train(train_loader, val_loader)
    print(metrics)


if __name__ == "__main__":
    main()