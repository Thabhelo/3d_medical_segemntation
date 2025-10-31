from __future__ import annotations

from pathlib import Path
import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import Trainer
from src.models.factory import create_model
from src.models.losses import get_loss


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=10, in_channels=1, num_classes=2, spatial_size=32):
        self.size = size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.spatial_size = spatial_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = torch.randn(self.in_channels, self.spatial_size, self.spatial_size, self.spatial_size)
        label = torch.randint(0, self.num_classes, (1, self.spatial_size, self.spatial_size, self.spatial_size))
        return {"image": image, "label": label}


class TestTrainer:
    def test_trainer_creation(self, tmp_path):
        model = create_model(architecture="unet", in_channels=1, out_channels=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = get_loss("dice")
        device = torch.device("cpu")

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            output_dir=tmp_path,
            max_epochs=2,
            num_classes=2,
        )
        assert trainer is not None
        assert trainer.max_epochs == 2

    def test_trainer_single_epoch(self, tmp_path):
        model = create_model(architecture="unet", in_channels=1, out_channels=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = get_loss("dice")
        device = torch.device("cpu")

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            output_dir=tmp_path,
            max_epochs=1,
            num_classes=2,
            amp=False,
        )

        train_dataset = DummyDataset(size=4, spatial_size=32)
        val_dataset = DummyDataset(size=2, spatial_size=32)
        train_loader = DataLoader(train_dataset, batch_size=2)
        val_loader = DataLoader(val_dataset, batch_size=1)

        metrics = trainer.train(train_loader, val_loader)
        assert "best_dice" in metrics
        assert isinstance(metrics["best_dice"], float)

    def test_checkpoint_saved(self, tmp_path):
        model = create_model(architecture="unet", in_channels=1, out_channels=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = get_loss("dice")
        device = torch.device("cpu")

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            output_dir=tmp_path,
            max_epochs=1,
            num_classes=2,
            amp=False,
        )

        train_dataset = DummyDataset(size=4, spatial_size=32)
        val_dataset = DummyDataset(size=2, spatial_size=32)
        train_loader = DataLoader(train_dataset, batch_size=2)
        val_loader = DataLoader(val_dataset, batch_size=1)

        trainer.train(train_loader, val_loader)
        assert (tmp_path / "best.pth").exists()
        assert (tmp_path / "train.log").exists()

    def test_checkpoint_loading(self, tmp_path):
        model = create_model(architecture="unet", in_channels=1, out_channels=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=get_loss("dice"),
            device=torch.device("cpu"),
            output_dir=tmp_path,
            max_epochs=1,
            num_classes=2,
        )
        trainer._save_checkpoint(epoch=1, val_dice=0.5, best=True)

        checkpoint = torch.load(tmp_path / "best.pth", map_location="cpu")
        assert "model" in checkpoint
        assert "optimizer" in checkpoint
        assert "epoch" in checkpoint
        assert checkpoint["epoch"] == 1

    def test_validate_method(self, tmp_path):
        model = create_model(architecture="unet", in_channels=1, out_channels=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=get_loss("dice"),
            device=torch.device("cpu"),
            output_dir=tmp_path,
            max_epochs=1,
            num_classes=2,
            amp=False,
        )

        val_dataset = DummyDataset(size=2, spatial_size=32)
        val_loader = DataLoader(val_dataset, batch_size=1)

        dice_score = trainer.validate(val_loader)
        assert isinstance(dice_score, float)
        assert 0.0 <= dice_score <= 1.0

    def test_history_saved(self, tmp_path):
        model = create_model(architecture="unet", in_channels=1, out_channels=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=get_loss("dice"),
            device=torch.device("cpu"),
            output_dir=tmp_path,
            max_epochs=2,
            num_classes=2,
            amp=False,
        )

        train_dataset = DummyDataset(size=4, spatial_size=32)
        val_dataset = DummyDataset(size=2, spatial_size=32)
        train_loader = DataLoader(train_dataset, batch_size=2)
        val_loader = DataLoader(val_dataset, batch_size=1)

        trainer.train(train_loader, val_loader)
        assert (tmp_path / "history.json").exists()

        import json
        with open(tmp_path / "history.json") as f:
            history = json.load(f)
        assert len(history) == 2
        assert "epoch" in history[0]
        assert "train_loss" in history[0]
        assert "val_dice" in history[0]
