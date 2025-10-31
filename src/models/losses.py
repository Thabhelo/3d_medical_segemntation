from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
from monai.losses import DiceLoss, DiceCELoss, FocalLoss, TverskyLoss


class DiceCECombinedLoss(nn.Module):
    """Fallback combined loss when DiceCELoss doesn't support class weights.

    - Uses MONAI DiceLoss for overlap (multi-class compatible).
    - Uses torch.nn.CrossEntropyLoss for class-weighted CE.

    Class weights are registered as a buffer to follow the module's device/casting.
    """

    def __init__(
        self,
        ce_weight: Optional[Sequence[float]] = None,
        dice_kwargs: Optional[dict[str, Any]] = None,
        ce_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.dice = DiceLoss(sigmoid=True, **(dice_kwargs or {}))

        weight_tensor: Optional[torch.Tensor] = None
        if ce_weight is not None:
            weight_tensor = torch.as_tensor(ce_weight, dtype=torch.float32)
        # Register as buffer so it moves with .to(device) and works with AMP
        self.register_buffer("ce_weight", weight_tensor)  # type: ignore[arg-type]

        self.ce = nn.CrossEntropyLoss(weight=self.ce_weight, **(ce_kwargs or {}))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # logits: [B, C, D, H, W]
        # target can be:
        #   - class indices: [B, D, H, W] or [B, 1, D, H, W]
        #   - one-hot: [B, C, D, H, W]

        # Derive class indices for CE and one-hot for Dice, regardless of input format
        if target.ndim == logits.ndim and target.shape[1] == logits.shape[1]:
            # One-hot target provided
            target_onehot = target
            target_indices = target.argmax(dim=1)
        else:
            # Indices provided (possibly with channel dim = 1)
            if target.ndim == logits.ndim:
                target_indices = target.squeeze(1).long()
            elif target.ndim == logits.ndim - 1:
                target_indices = target.long()
            else:
                raise ValueError(f"Unexpected target shape for CE/Dice combo: {tuple(target.shape)}")
            # Build one-hot for Dice to match logits channels
            target_onehot = torch.nn.functional.one_hot(
                target_indices.clamp_min(0), num_classes=logits.shape[1]
            ).permute(0, 4, 1, 2, 3).to(logits.dtype)

        ce_loss = self.ce(logits, target_indices)
        dice_loss = self.dice(logits, target_onehot)
        return dice_loss + ce_loss


def get_loss(loss_name: str, class_weights: Optional[Sequence[float]] = None, **kwargs: Any) -> nn.Module:
    name = (loss_name or "dice_ce").strip().lower()
    if name == "dice":
        return DiceLoss(sigmoid=True, **kwargs)
    if name in {"dice_ce", "dice+ce"}:
        return DiceCELoss(sigmoid=True, **kwargs)
    if name == "dice_ce_balanced":
        ce_weights_list = class_weights or [1.0, 1.0, 3.0]
        try:
            return DiceCELoss(sigmoid=True, ce_weight=ce_weights_list, **kwargs)  # type: ignore[arg-type]
        except TypeError:
            return DiceCECombinedLoss(ce_weight=ce_weights_list)
    if name == "focal":
        return FocalLoss(**kwargs)
    if name == "tversky":
        return TverskyLoss(**kwargs)
    raise ValueError(f"Unknown loss: {loss_name}")


