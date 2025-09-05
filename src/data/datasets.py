from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from .utils import fuse_totalseg_segmentations


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
        # Robustly discover case directories anywhere under root by looking for *_seg(.nii|.nii.gz)
        seg_files = [
            p
            for p in self.root_dir.rglob("*_seg.nii*")
            if p.suffix in {".nii", ".gz"} or p.name.endswith(".nii.gz")
        ]
        case_dirs = sorted({p.parent for p in seg_files})
        return list(case_dirs)

    def get_data_dicts(self) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        # Accept common BraTS modality aliases (BraTS 2021/2024):
        # t1: [t1, t1n]; t1ce: [t1ce, t1c]; t2: [t2, t2w]; flair: [flair, t2f]
        modality_aliases = {
            "t1": ["t1n", "t1"],
            "t1ce": ["t1c", "t1ce"],
            "t2": ["t2w", "t2"],
            "flair": ["t2f", "flair"],
        }

        for case_dir in self.cases:
            case_id = case_dir.name

            def find_file_for_mod(mod: str) -> Optional[Path]:
                tokens = modality_aliases.get(mod, [mod])
                matches: List[Path] = []
                for f in case_dir.iterdir():
                    if not f.is_file():
                        continue
                    name = f.name.lower()
                    if (name.endswith(".nii") or name.endswith(".nii.gz")) and any(t in name for t in tokens):
                        matches.append(f)
                # Prefer .nii.gz if multiple
                matches.sort(key=lambda p: (0 if p.name.endswith(".nii.gz") else 1, p.name))
                return matches[0] if matches else None

            images: List[str] = []
            for mod in self.MODALITIES:
                f = find_file_for_mod(mod)
                if f is None:
                    images = []
                    break
                images.append(f.as_posix())

            if not images:
                continue

            # find label (segmentation) file
            label = None
            for f in case_dir.iterdir():
                if not f.is_file():
                    continue
                n = f.name.lower()
                if (n.endswith(".nii") or n.endswith(".nii.gz")) and ("_seg" in n or n == "seg.nii.gz" or n == "seg.nii"):
                    label = f
                    break
            if label is None:
                continue

            data.append({
                "image": images,
                "label": label.as_posix(),
                "case_id": case_id,
            })

        return data


class MSDLiverDataset(MedicalDataset):
    """
    Medical Segmentation Decathlon - Task03 Liver.

    Expected layout (typical MSD):
      root_dir/
        imagesTr/*.nii.gz
        labelsTr/*.nii.gz

    This implementation is robust to slight naming variations:
    - Images may end with `_0000.nii.gz` (single modality); labels typically do not.
    - Falls back to matching stems when `_0000` is absent.
    """

    def __init__(self, root_dir: str, split: str = "train", transforms: Optional[Any] = None):
        super().__init__(root_dir=root_dir, split=split, transforms=transforms)
        self.images_dir, self.labels_dir = self._resolve_directories()

    @property
    def num_classes(self) -> int:
        # Background, Liver, Tumor
        return 3

    def get_class_info(self) -> Dict[str, Any]:
        return {
            "labels": {
                0: "background",
                1: "liver",
                2: "tumor",
            }
        }

    def _resolve_directories(self) -> tuple[Path, Path]:
        candidates_img = [
            self.root_dir / "imagesTr",
            self.root_dir / "images",
            self.root_dir / "img",
            self.root_dir / "training" / "images",
        ]
        candidates_lbl = [
            self.root_dir / "labelsTr",
            self.root_dir / "labels",
            self.root_dir / "lab",
            self.root_dir / "training" / "labels",
        ]

        images_dir = next((p for p in candidates_img if p.exists()), self.root_dir)
        labels_dir = next((p for p in candidates_lbl if p.exists()), self.root_dir)
        return images_dir, labels_dir

    def get_data_dicts(self) -> List[Dict[str, Any]]:
        if not self.images_dir.exists() or not self.labels_dir.exists():
            return []

        def is_nifti(p: Path) -> bool:
            return p.is_file() and (p.suffix in {".nii", ".gz"} or p.name.endswith(".nii.gz"))

        images = sorted([p for p in self.images_dir.iterdir() if is_nifti(p)])
        data: List[Dict[str, Any]] = []

        for img in images:
            stem = img.name
            if stem.endswith("_0000.nii.gz"):
                base = stem.replace("_0000.nii.gz", "")
                label_candidates = [
                    self.labels_dir / f"{base}.nii.gz",
                    self.labels_dir / f"{base}.nii",
                ]
            elif stem.endswith(".nii.gz"):
                base = stem.replace(".nii.gz", "")
                label_candidates = [
                    self.labels_dir / f"{base}.nii.gz",
                    self.labels_dir / f"{base}.nii",
                ]
            elif stem.endswith(".nii"):
                base = stem.replace(".nii", "")
                label_candidates = [
                    self.labels_dir / f"{base}.nii.gz",
                    self.labels_dir / f"{base}.nii",
                ]
            else:
                label_candidates = []

            label_path = next((p for p in label_candidates if p.exists()), None)
            if label_path is None:
                # Try a fuzzy match by stem containment
                base_stem = Path(base).name if 'base' in locals() else img.stem
                for cand in self.labels_dir.iterdir():
                    if not is_nifti(cand):
                        continue
                    if base_stem in cand.stem or cand.stem in base_stem:
                        label_path = cand
                        break

            if label_path is None:
                continue

            data.append({
                "image": img.as_posix(),
                "label": label_path.as_posix(),
                "case_id": Path(img.stem.replace("_0000", "")).name,
            })

        return data


class TotalSegmentatorDataset(MedicalDataset):
    """
    TotalSegmentator dataset loader.

    Assumes a simple images/labels directory layout. Labels contain multiple
    anatomical classes encoded as integers in a single volume.
    """

    def __init__(self, root_dir: str, split: str = "train", transforms: Optional[Any] = None):
        super().__init__(root_dir=root_dir, split=split, transforms=transforms)
        self.images_dir, self.labels_dir = self._resolve_directories()

    @property
    def num_classes(self) -> int:
        # Background + 117 anatomical structures
        return 118

    def get_class_info(self) -> Dict[str, Any]:
        return {
            "num_classes": 118,
            "description": "Background + 117 anatomical structures",
        }

    def _resolve_directories(self) -> tuple[Path, Path]:
        candidates_img = [
            self.root_dir / "imagesTr",
            self.root_dir / "images",
            self.root_dir / "ct",
            self.root_dir / "training" / "images",
        ]
        candidates_lbl = [
            self.root_dir / "labelsTr",
            self.root_dir / "labels",
            self.root_dir / "segmentations",
            self.root_dir / "training" / "labels",
        ]

        images_dir = next((p for p in candidates_img if p.exists()), self.root_dir)
        labels_dir = next((p for p in candidates_lbl if p.exists()), self.root_dir)
        return images_dir, labels_dir

    def get_data_dicts(self) -> List[Dict[str, Any]]:
        if not self.images_dir.exists() or not self.labels_dir.exists():
            return []

        def is_nifti(p: Path) -> bool:
            return p.is_file() and (p.suffix in {".nii", ".gz"} or p.name.endswith(".nii.gz"))

        images = sorted([p for p in self.images_dir.iterdir() if is_nifti(p)])
        data: List[Dict[str, Any]] = []

        for img in images:
            base_stem = img.parent.name if img.name == "ct.nii.gz" else img.stem.replace("_0000", "")

            # When dataset is laid out as sXXXX/{ct.nii.gz,segmentations/*.nii.gz}
            # create a fused label map per subject under the subject dir
            subject_dir = img.parent if img.name == "ct.nii.gz" else self.labels_dir / base_stem
            try:
                fused = fuse_totalseg_segmentations(subject_dir)
            except Exception:
                # Fallback: try label files matching by stem
                candidates = [
                    self.labels_dir / f"{base_stem}.nii.gz",
                    self.labels_dir / f"{base_stem}.nii",
                ]
                fused = next((p for p in candidates if p.exists()), None)

            if fused is None or not Path(fused).exists():
                continue

            image_path = img if img.name == "ct.nii.gz" else (self.images_dir / f"{base_stem}.nii.gz")
            if not image_path.exists():
                # try subject_dir/ct.nii.gz
                alt = subject_dir / "ct.nii.gz"
                if alt.exists():
                    image_path = alt
                else:
                    continue

            data.append({
                "image": image_path.as_posix(),
                "label": str(fused),
                "case_id": base_stem,
            })

        return data


