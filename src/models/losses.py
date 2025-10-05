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
        # target: [B, 1, D, H, W] or [B, D, H, W]
        if target.ndim == logits.ndim:
            target = target.squeeze(1)
        ce_loss = self.ce(logits, target.long())
        dice_loss = self.dice(logits, target)
        return dice_loss + ce_loss


def get_loss(loss_name: str, **kwargs: Any) -> nn.Module:
    name = (loss_name or "dice_ce").strip().lower()
    if name == "dice":
        return DiceLoss(sigmoid=True, **kwargs)
    if name in {"dice_ce", "dice+ce"}:
        return DiceCELoss(sigmoid=True, **kwargs)
    if name == "dice_ce_balanced":
        # Prefer MONAI DiceCELoss with weights when supported by the installed version.
        # Fall back to a custom Dice + CE combination if the signature differs.
        ce_weights_list = [1.0, 1.0, 3.0]  # [background, liver, tumor]
        try:
            # Some MONAI versions expect `ce_weight`, others may differ.
            return DiceCELoss(sigmoid=True, ce_weight=ce_weights_list, **kwargs)  # type: ignore[arg-type]
        except TypeError:
            # Fallback: explicit combination with torch.nn.CrossEntropyLoss
            return DiceCECombinedLoss(ce_weight=ce_weights_list)
    if name == "focal":
        return FocalLoss(**kwargs)
    if name == "tversky":
        return TverskyLoss(**kwargs)
    raise ValueError(f"Unknown loss: {loss_name}")


