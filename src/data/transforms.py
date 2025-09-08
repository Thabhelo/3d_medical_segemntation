from __future__ import annotations

from typing import Optional, Tuple

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    NormalizeIntensityd,
    CropForegroundd,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureTyped,
)


def get_preprocessing_transforms(dataset_name: str, phase: str) -> Compose:
    """
    Return preprocessing transforms for a dataset and phase.

    dataset_name: one of {"brats", "msd_liver", "totalsegmentator"}
    phase: "train" | "val" | "test"
    """
    name = dataset_name.lower().replace(" ", "_")
    if name in {"brats2021"}:
        name = "brats"
    if name in {"msd", "task03_liver"}:
        name = "msd_liver"
    if name in {"totalsegmentator"}:
        name = "totalsegmentator"

    base = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0)),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ]

    if name == "brats":
        ds_specific = [
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
    elif name == "msd_liver":
        ds_specific = [
            ScaleIntensityRanged(
                keys="image", a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
    elif name == "totalsegmentator":
        ds_specific = [
            ScaleIntensityRanged(
                keys="image", a_min=-1024, a_max=1024, b_min=0.0, b_max=1.0, clip=True
            )
        ]
    else:
        ds_specific = []

    return Compose(base + ds_specific)


def get_augmentation_transforms(patch_size: Tuple[int, int, int]) -> Compose:
    """Training-time augmentation transforms."""
    from monai.transforms import SpatialPadd, RandSpatialCropd
    
    return Compose(
        [
            # Ensure minimum size before cropping
            SpatialPadd(keys=["image", "label"], spatial_size=patch_size, mode="constant"),
            RandSpatialCropd(keys=["image", "label"], roi_size=patch_size, random_size=False),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_transforms(dataset_name: str, phase: str, patch_size: Optional[Tuple[int, int, int]] = None) -> Compose:
    """Convenience to build full transforms given phase.

    For training, returns preprocessing + augmentation; for val/test, preprocessing only.
    """
    preprocess = get_preprocessing_transforms(dataset_name, phase)
    if phase.lower() == "train" and patch_size is not None:
        return Compose([preprocess, get_augmentation_transforms(patch_size)])
    return preprocess


