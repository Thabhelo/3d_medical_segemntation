from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, TypeVar
import random

from monai.data import CacheDataset, Dataset
from torch.utils.data import DataLoader

from .datasets import BraTSDataset, MSDLiverDataset, TotalSegmentatorDataset
from .transforms import get_transforms


T = TypeVar("T")


def deterministic_split(
    items: Sequence[T], train_fraction: float = 0.8, seed: int = 42
) -> Tuple[List[T], List[T]]:
    """
    Deterministically split a sequence into train and validation subsets.

    Parameters
    ----------
    items: Sequence[T]
        Input items to split.
    train_fraction: float
        Fraction of items assigned to the training split.
    seed: int
        Random seed to ensure deterministic behavior.

    Returns
    -------
    (train, val): Tuple[List[T], List[T]]
        Two lists with a deterministic split of the input items.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")

    indices = list(range(len(items)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    cutoff = int(len(indices) * train_fraction)
    train_idx = indices[:cutoff]
    val_idx = indices[cutoff:]

    train = [items[i] for i in train_idx]
    val = [items[i] for i in val_idx]
    return train, val


def find_case_directories(root_dir: Path) -> List[Path]:
    """
    Find case directories under a dataset root. A "case directory" is defined
    as a directory that contains at least one NIfTI file (e.g., .nii or .nii.gz).

    This is a heuristic intended to work across different dataset layouts.
    """
    if not root_dir.exists():
        return []

    nifti_exts = {".nii", ".nii.gz"}
    case_dirs: List[Path] = []

    for path in root_dir.rglob("*"):
        if path.is_dir():
            has_nifti = any(
                f.is_file() and (f.suffix in nifti_exts or ".nii.gz" in f.name)
                for f in path.iterdir()
            )
            if has_nifti:
                case_dirs.append(path)
    return case_dirs


def normalize_dataset_name(name: str) -> str:
    """Map user-facing names to internal canonical dataset keys."""
    n = name.strip().lower().replace(" ", "_")
    aliases = {
        "brats": "brats",
        "brats2021": "brats",
        "msd": "msd_liver",
        "msd_liver": "msd_liver",
        "task03_liver": "msd_liver",
        "totalsegmentator": "totalsegmentator",
    }
    return aliases.get(n, n)


## Note: TotalSegmentator fusion utility lives in src/data/postprocess.py to avoid circular imports.

def get_dataset_instance(
    dataset_name: str,
    root_dir: str,
    split: str,
    transforms=None,
):
    name = normalize_dataset_name(dataset_name)
    if name == "brats":
        return BraTSDataset(root_dir=root_dir, split=split, transforms=transforms)
    if name == "msd_liver":
        return MSDLiverDataset(root_dir=root_dir, split=split, transforms=transforms)
    if name == "totalsegmentator":
        return TotalSegmentatorDataset(root_dir=root_dir, split=split, transforms=transforms)
    raise ValueError(f"Unknown dataset: {dataset_name}")


def create_monai_datasets(
    dataset_name: str,
    root_dir: str,
    train_fraction: float = 0.8,
    seed: int = 42,
    patch_size: Optional[Tuple[int, int, int]] = (128, 128, 128),
    cache_rate: float = 0.0,
):
    """
    Build MONAI Datasets (train/val) with appropriate transforms.
    """
    # Build transforms
    norm_name = normalize_dataset_name(dataset_name)
    train_transforms = get_transforms(norm_name, phase="train", patch_size=patch_size)
    val_transforms = get_transforms(norm_name, phase="val", patch_size=patch_size)

    # Instantiate dataset class and obtain dicts
    ds = get_dataset_instance(norm_name, root_dir, split="train")
    dicts = ds.get_data_dicts()
    train_dicts, val_dicts = deterministic_split(dicts, train_fraction=train_fraction, seed=seed)

    if cache_rate and cache_rate > 0.0:
        train_ds = CacheDataset(data=train_dicts, transform=train_transforms, cache_rate=cache_rate)
        val_ds = CacheDataset(data=val_dicts, transform=val_transforms, cache_rate=cache_rate)
    else:
        train_ds = Dataset(data=train_dicts, transform=train_transforms)
        val_ds = Dataset(data=val_dicts, transform=val_transforms)

    return train_ds, val_ds


def create_dataloaders(
    dataset_name: str,
    root_dir: str,
    batch_size: int = 2,
    num_workers: int = 4,
    train_fraction: float = 0.8,
    seed: int = 42,
    patch_size: Optional[Tuple[int, int, int]] = (128, 128, 128),
    cache_rate: float = 0.0,
):
    train_ds, val_ds = create_monai_datasets(
        dataset_name=dataset_name,
        root_dir=root_dir,
        train_fraction=train_fraction,
        seed=seed,
        patch_size=patch_size,
        cache_rate=cache_rate,
    )

    persistent = num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )
    return train_loader, val_loader


