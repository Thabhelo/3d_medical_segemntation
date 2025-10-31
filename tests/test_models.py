from __future__ import annotations

import torch
import pytest

from src.models.factory import create_model
from src.models.losses import get_loss, DiceCECombinedLoss


class TestModelFactory:
    def test_create_unet(self):
        model = create_model(architecture="unet", in_channels=1, out_channels=2)
        assert model is not None
        assert hasattr(model, "forward")

    @pytest.mark.skipif(True, reason="UNETR requires einops (optional dependency)")
    def test_create_unetr(self):
        model = create_model(architecture="unetr", in_channels=1, out_channels=2, img_size=(96, 96, 96))
        assert model is not None

    def test_create_segresnet(self):
        model = create_model(architecture="segresnet", in_channels=1, out_channels=2)
        assert model is not None

    def test_invalid_architecture(self):
        with pytest.raises(ValueError):
            create_model(architecture="invalid_arch", in_channels=1, out_channels=2)

    def test_model_forward_unet(self):
        model = create_model(architecture="unet", in_channels=1, out_channels=2)
        x = torch.randn(1, 1, 32, 32, 32)
        output = model(x)
        assert output.shape[0] == 1
        assert output.shape[1] == 2

    def test_model_forward_segresnet(self):
        model = create_model(architecture="segresnet", in_channels=1, out_channels=3)
        x = torch.randn(1, 1, 32, 32, 32)
        output = model(x)
        assert output.shape[0] == 1
        assert output.shape[1] == 3


class TestLossFunctions:
    def test_get_dice_loss(self):
        loss_fn = get_loss("dice")
        assert loss_fn is not None

    def test_get_dice_ce_loss(self):
        loss_fn = get_loss("dice_ce")
        assert loss_fn is not None

    def test_get_dice_ce_balanced_loss(self):
        loss_fn = get_loss("dice_ce_balanced")
        assert loss_fn is not None

    def test_get_focal_loss(self):
        loss_fn = get_loss("focal")
        assert loss_fn is not None

    def test_dice_ce_balanced_with_custom_weights(self):
        loss_fn = get_loss("dice_ce_balanced", class_weights=[1.0, 2.0, 3.0])
        assert loss_fn is not None

    def test_invalid_loss_name(self):
        with pytest.raises(ValueError):
            get_loss("invalid_loss")

    def test_dice_ce_combined_loss_forward(self):
        loss_fn = DiceCECombinedLoss(ce_weight=[1.0, 1.0, 2.0])
        logits = torch.randn(2, 3, 8, 8, 8)
        labels = torch.randint(0, 3, (2, 1, 8, 8, 8))
        loss = loss_fn(logits, labels)
        assert loss.item() >= 0
        assert not torch.isnan(loss)


class TestModelOutputShapes:
    @pytest.mark.parametrize("out_channels", [2, 3, 4])
    def test_unet_output_channels(self, out_channels):
        model = create_model(architecture="unet", in_channels=1, out_channels=out_channels)
        x = torch.randn(1, 1, 32, 32, 32)
        output = model(x)
        assert output.shape[1] == out_channels

    @pytest.mark.parametrize("in_channels", [1, 4])
    def test_unet_input_channels(self, in_channels):
        model = create_model(architecture="unet", in_channels=in_channels, out_channels=2)
        x = torch.randn(1, in_channels, 32, 32, 32)
        output = model(x)
        assert output.shape[0] == 1
