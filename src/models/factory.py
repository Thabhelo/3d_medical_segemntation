from __future__ import annotations

from typing import Any

import torch.nn as nn

from .architectures import build_unet, build_unetr, build_segresnet


def create_model(
    *,
    architecture: str,
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 2,
    **kwargs: Any,
) -> nn.Module:
    name = architecture.strip().lower()
    if name == "unet":
        return build_unet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs,
        )
    if name == "unetr":
        return build_unetr(
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs,
        )
    if name == "segresnet":
        return build_segresnet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs,
        )
    raise ValueError(f"Unknown architecture: {architecture}")


