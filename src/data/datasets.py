from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DatasetSplit:
    train: List[Dict[str, Any]]
    val: List[Dict[str, Any]]


class MedicalDataset(ABC):
    """Abstract base class for medical datasets returning MONAI-style dicts."""

    def __init__(self, root_dir: str, split: str, transforms: Optional[Any] = None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transforms = transforms

    @abstractmethod
    def get_data_dicts(self) -> List[Dict[str, Any]]:
        """Return list of data dictionaries with image/label paths."""
        raise NotImplementedError

    @abstractmethod
    def get_class_info(self) -> Dict[str, Any]:
        """Return class labels and descriptions."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_classes(self) -> int:
        raise NotImplementedError


class BraTSDataset(MedicalDataset):
    """
    BraTS-style dataset loader.

    Expected directory layout (example; variations exist):
      root_dir/
        PatientID/
          PatientID_t1.nii.gz
          PatientID_t1ce.nii.gz
          PatientID_t2.nii.gz
          PatientID_flair.nii.gz
          PatientID_seg.nii.gz
    """

    MODALITIES = ("t1", "t1ce", "t2", "flair")

    def __init__(self, root_dir: str, split: str = "train", transforms: Optional[Any] = None):
        super().__init__(root_dir=root_dir, split=split, transforms=transforms)
        self.cases = self._discover_cases()

    @property
    def num_classes(self) -> int:
        # Background + 3 tumor classes (NCR/NET, ED, ET) typically 4 channels
        return 4

    def get_class_info(self) -> Dict[str, Any]:
        return {
            "labels": {
                0: "background",
                1: "ncr_net",
                2: "ed",
                3: "et",
            }
        }

    def _discover_cases(self) -> List[Path]:
        if not self.root_dir.exists():
            return []
        # A case directory is one that contains a segmentation file "*_seg.nii.gz"
        cases: List[Path] = []
        for path in self.root_dir.iterdir():
            if path.is_dir():
                seg = next(path.glob("*_seg.nii.gz"), None)
                if seg is not None:
                    cases.append(path)
        cases.sort()
        return cases

    def get_data_dicts(self) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        for case_dir in self.cases:
            case_id = case_dir.name
            images = []
            for mod in self.MODALITIES:
                candidate = next(case_dir.glob(f"*_{mod}.nii.gz"), None)
                if candidate is None:
                    # If any modality missing, skip the case to avoid runtime failures
                    images = []
                    break
                images.append(candidate.as_posix())

            if not images:
                continue

            label = next(case_dir.glob("*_seg.nii.gz"), None)
            if label is None:
                continue

            data.append({
                "image": images,
                "label": label.as_posix(),
                "case_id": case_id,
            })

        return data


