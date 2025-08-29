from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, TypeVar
import random


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


