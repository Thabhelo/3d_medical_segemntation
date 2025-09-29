from __future__ import annotations

from typing import Any

import torch.nn as nn
from monai.networks.nets import UNet, UNETR, SegResNet, BasicUNet


def build_unet(
    *,
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 2,
    features: tuple[int, int, int, int, int, int] = (32, 32, 64, 128, 256, 32),
    act: str = "relu",
    norm: str = "batch",
    dropout: float = 0.1,
    **kwargs: Any,
) -> nn.Module:
    # Use BasicUNet which handles arbitrary input sizes better
    return BasicUNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        act=act,
        norm=norm,
        dropout=dropout,
        **kwargs,
    )


def build_unetr(
    *,
    in_channels: int = 1,
    out_channels: int = 2,
    img_size: tuple[int, int, int] = (128, 128, 128),
    feature_size: int = 16,
    hidden_size: int = 768,
    mlp_dim: int = 3072,
    num_heads: int = 12,
    proj_type: str = "perceptron",
    norm_name: str = "instance",
    res_block: bool = True,
    dropout_rate: float = 0.0,
    **kwargs: Any,
) -> nn.Module:
    return UNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        feature_size=feature_size,
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        proj_type=proj_type,
        norm_name=norm_name,
        res_block=res_block,
        dropout_rate=dropout_rate,
        **kwargs,
    )


def build_segresnet(
    *,
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 2,
    init_filters: int = 32,
    dropout_prob: float = 0.2,
    act: str = "relu",
    norm: str = "instance",
    use_conv_final: bool = True,
    blocks_down: list[int] | tuple[int, ...] = (1, 2, 2, 4),
    blocks_up: list[int] | tuple[int, ...] = (1, 1, 1),
    **kwargs: Any,
) -> nn.Module:
    return SegResNet(
        spatial_dims=spatial_dims,
        init_filters=init_filters,
        in_channels=in_channels,
        out_channels=out_channels,
        dropout_prob=dropout_prob,
        act=act,
        # MONAI SegResNet uses norm_name for normalization selection in recent versions
        norm_name=norm,
        use_conv_final=use_conv_final,
        blocks_down=list(blocks_down),
        blocks_up=list(blocks_up),
        **kwargs,
    )


