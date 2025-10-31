from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import nibabel as nib
import pytest

from src.data.datasets import BraTSDataset, MSDLiverDataset, TotalSegmentatorDataset
from src.data.utils import deterministic_split, create_monai_datasets
from src.data.transforms import get_transforms


def create_dummy_nifti(path: Path, shape=(10, 10, 10), dtype=np.float32):
    """Create a dummy NIfTI file for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.random.rand(*shape).astype(dtype)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(img, str(path))


class TestDeterministicSplit:
    def test_split_reproducible(self):
        items = list(range(100))
        train1, val1 = deterministic_split(items, train_fraction=0.8, seed=42)
        train2, val2 = deterministic_split(items, train_fraction=0.8, seed=42)
        assert train1 == train2
        assert val1 == val2

    def test_split_ratio(self):
        items = list(range(100))
        train, val = deterministic_split(items, train_fraction=0.8, seed=42)
        assert len(train) == 80
        assert len(val) == 20

    def test_split_no_overlap(self):
        items = list(range(50))
        train, val = deterministic_split(items, train_fraction=0.7, seed=42)
        assert len(set(train) & set(val)) == 0

    def test_split_invalid_fraction(self):
        items = list(range(10))
        with pytest.raises(ValueError):
            deterministic_split(items, train_fraction=1.5)


class TestBraTSDataset:
    def test_dataset_creation_empty(self, tmp_path):
        dataset = BraTSDataset(root_dir=str(tmp_path), split="train")
        data_dicts = dataset.get_data_dicts()
        assert len(data_dicts) == 0

    def test_num_classes(self):
        dataset = BraTSDataset(root_dir="/tmp", split="train")
        assert dataset.num_classes == 4

    def test_class_info(self):
        dataset = BraTSDataset(root_dir="/tmp", split="train")
        info = dataset.get_class_info()
        assert "labels" in info
        assert len(info["labels"]) == 4


class TestMSDLiverDataset:
    def test_dataset_creation_empty(self, tmp_path):
        dataset = MSDLiverDataset(root_dir=str(tmp_path), split="train")
        data_dicts = dataset.get_data_dicts()
        assert len(data_dicts) == 0

    def test_num_classes(self):
        dataset = MSDLiverDataset(root_dir="/tmp", split="train")
        assert dataset.num_classes == 3

    def test_with_dummy_data(self, tmp_path):
        images_dir = tmp_path / "imagesTr"
        labels_dir = tmp_path / "labelsTr"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        create_dummy_nifti(images_dir / "liver_0_0000.nii.gz")
        create_dummy_nifti(labels_dir / "liver_0.nii.gz", dtype=np.uint8)

        dataset = MSDLiverDataset(root_dir=str(tmp_path), split="train")
        data_dicts = dataset.get_data_dicts()
        assert len(data_dicts) == 1
        assert "image" in data_dicts[0]
        assert "label" in data_dicts[0]


class TestTotalSegmentatorDataset:
    def test_num_classes(self):
        dataset = TotalSegmentatorDataset(root_dir="/tmp", split="train")
        assert dataset.num_classes == 118


class TestTransforms:
    def test_brats_transforms(self):
        transforms = get_transforms("brats", phase="train", patch_size=(64, 64, 64))
        assert transforms is not None

    def test_msd_liver_transforms(self):
        transforms = get_transforms("msd_liver", phase="val", patch_size=(64, 64, 64))
        assert transforms is not None

    def test_transforms_no_patch_size(self):
        transforms = get_transforms("brats", phase="val", patch_size=None)
        assert transforms is not None
