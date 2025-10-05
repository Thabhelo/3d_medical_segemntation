from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from monai.losses import DiceLoss, DiceCELoss, FocalLoss, TverskyLoss


def get_loss(loss_name: str, **kwargs: Any) -> nn.Module:
    name = (loss_name or "dice_ce").strip().lower()
    if name == "dice":
        return DiceLoss(sigmoid=True, **kwargs)
    if name in {"dice_ce", "dice+ce"}:
        return DiceCELoss(sigmoid=True, **kwargs)
    if name == "dice_ce_balanced":
        # Class-balanced loss for MSD Liver (higher weight on tumor class)
        return DiceCELoss(
            sigmoid=True,
            ce_weight=torch.tensor([1.0, 1.0, 3.0]),  # [bg, liver, tumor] - 3x weight on tumor
            **kwargs
        )
    if name == "focal":
        return FocalLoss(**kwargs)
    if name == "tversky":
        return TverskyLoss(**kwargs)
    raise ValueError(f"Unknown loss: {loss_name}")


