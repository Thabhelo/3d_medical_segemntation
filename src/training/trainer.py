from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        device: torch.device,
        output_dir: Path,
        max_epochs: int = 2,
        amp: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_epochs = max_epochs
        self.amp = amp

        self.post_pred = AsDiscrete(argmax=True, to_onehot=None)
        self.post_label = AsDiscrete(to_onehot=None)
        self.val_dice = DiceMetric(include_background=False, reduction="mean")

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, float]:
        scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        best_dice = -1.0

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            running_loss = 0.0
            for batch in train_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.amp):
                    logits = self.model(images)
                    loss = self.loss_fn(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                running_loss += float(loss.item())

            train_loss = running_loss / max(1, len(train_loader))
            val_dice = self.validate(val_loader)

            if val_dice > best_dice:
                best_dice = val_dice
                self._save_checkpoint(epoch, best=True)

            self._save_checkpoint(epoch, best=False)

        return {"best_dice": best_dice}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        self.val_dice.reset()
        for batch in val_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            logits = self.model(images)

            # convert to onehot/discrete lists for metric
            y_pred = [self.post_pred(p) for p in decollate_batch(logits)]
            y = [self.post_label(p) for p in decollate_batch(labels)]
            self.val_dice(y_pred, y)

        return float(self.val_dice.aggregate().item())

    def _save_checkpoint(self, epoch: int, best: bool = False) -> None:
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
        }
        fname = "best.pth" if best else f"epoch_{epoch}.pth"
        torch.save(ckpt, self.output_dir / fname)


