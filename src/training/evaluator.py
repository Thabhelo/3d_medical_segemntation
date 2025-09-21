from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    model.to(device)
    post_pred = AsDiscrete(argmax=True)
    post_label = AsDiscrete()
    dice = DiceMetric(include_background=False, reduction="mean")
    dice.reset()
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(images)
        y_pred = [post_pred(p) for p in decollate_batch(logits)]
        y = [post_label(p) for p in decollate_batch(labels)]
        dice(y_pred, y)
    return {"dice": float(dice.aggregate().item())}


