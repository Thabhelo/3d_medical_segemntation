from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot
import time


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
        num_classes: int = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        save_every_epoch: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_epochs = max_epochs
        self.amp = amp and device.type == "cuda"  # Only use AMP on GPU
        self.num_classes = num_classes
        self.scheduler = scheduler
        self.save_every_epoch = save_every_epoch

        # Post-processing for metric computation
        # DiceMetric expects one-hot tensors [B, C, H, W, D]
        # We'll convert manually to avoid AsDiscrete batch dimension bugs
        self.val_dice = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, start_epoch: int = 1) -> Dict[str, float]:
        scaler = torch.amp.GradScaler('cuda', enabled=self.amp)
        best_dice = -1.0

        for epoch in range(start_epoch, self.max_epochs + 1):
            epoch_start = time.time()
            remaining_epochs = (self.max_epochs - epoch) + 1  # include current until we print ETA after validate
            print(f"[Epoch {epoch}/{self.max_epochs}] Starting training…", flush=True)
            self.model.train()
            running_loss = 0.0
            for batch_data in train_loader:
                # Handle MONAI batch format (can be list of dicts or dict)
                batch = batch_data[0] if isinstance(batch_data, list) else batch_data
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=self.amp):
                    logits = self.model(images)
                    # Convert labels to one-hot if needed for loss calculation
                    if labels.shape[1] == 1 and logits.shape[1] > 1:
                        # Ensure labels are integers and within valid range [0, num_classes-1]
                        labels_int = labels.long().clamp(0, logits.shape[1] - 1)
                        labels_onehot = one_hot(labels_int, num_classes=logits.shape[1])
                    else:
                        labels_onehot = labels
                    loss = self.loss_fn(logits, labels_onehot)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                running_loss += float(loss.item())

            train_loss = running_loss / max(1, len(train_loader))
            val_dice = self.validate(val_loader)
            elapsed = time.time() - epoch_start
            # Simple ETA based on this epoch duration times remaining epochs
            eta_seconds = max(0.0, elapsed * max(0, self.max_epochs - epoch))
            eta_min = int(eta_seconds // 60)
            eta_sec = int(eta_seconds % 60)
            print(
                f"[Epoch {epoch}/{self.max_epochs}] loss={train_loss:.4f} val_dice={val_dice:.4f} "
                f"time={elapsed:.1f}s ETA~{eta_min}m{eta_sec:02d}s",
                flush=True,
            )

            if val_dice > best_dice:
                best_dice = val_dice
                self._save_checkpoint(epoch, val_dice, best=True)

            # Save checkpoints based on configuration
            if self.save_every_epoch:
                self._save_checkpoint(epoch, val_dice, best=False)
            elif epoch % 10 == 0:
                self._save_checkpoint(epoch, val_dice, best=False)

            # Step the learning rate scheduler if available
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_dice)
                else:
                    self.scheduler.step()

        return {"best_dice": best_dice}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        self.val_dice.reset()
        for batch_data in val_loader:
            # Handle MONAI batch format (can be list of dicts or dict)
            batch = batch_data[0] if isinstance(batch_data, list) else batch_data
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            logits = self.model(images)

            # Convert to one-hot manually (avoid AsDiscrete batch dimension bugs)
            # Predictions: argmax to get class indices, then one-hot encode
            y_pred_indices = torch.argmax(logits, dim=1, keepdim=True)  # [B, 1, H, W, D]
            y_pred_onehot = one_hot(y_pred_indices, num_classes=self.num_classes)  # [B, C, H, W, D]
            
            # Labels: squeeze channel dim, ensure long dtype, then one-hot encode
            labels_squeezed = labels.squeeze(1).long()  # [B, H, W, D]
            y_true_onehot = one_hot(labels_squeezed.unsqueeze(1), num_classes=self.num_classes)  # [B, C, H, W, D]
            
            self.val_dice(y_pred_onehot, y_true_onehot)

        return float(self.val_dice.aggregate().item())

    def _save_checkpoint(self, epoch: int, val_dice: float, best: bool = False) -> None:
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "metrics": {"val_dice": val_dice},
        }
        # Save scheduler state if available
        if self.scheduler is not None:
            ckpt["scheduler"] = self.scheduler.state_dict()
        
        fname = "best.pth" if best else f"epoch_{epoch}.pth"
        torch.save(ckpt, self.output_dir / fname)


