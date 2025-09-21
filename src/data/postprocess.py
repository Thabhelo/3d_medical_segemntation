from __future__ import annotations

from pathlib import Path
import numpy as np
import nibabel as nib


def fuse_totalseg_segmentations(subject_dir: Path, output_name: str = "combined_labels.nii.gz") -> Path:
    """
    Combine TotalSegmentator per-structure masks (segmentations/*.nii.gz) into a single
    indexed label map (background=0, structures start at 1 in deterministic order).

    Returns the path to the fused NIfTI label file.
    """
    ct_path = subject_dir / "ct.nii.gz"
    seg_dir = subject_dir / "segmentations"
    out_path = subject_dir / output_name

    if out_path.exists():
        return out_path
    if not ct_path.exists() or not seg_dir.exists():
        raise FileNotFoundError(f"Missing ct.nii.gz or segmentations folder in {subject_dir}")

    ct_img = nib.load(str(ct_path))
    label = np.zeros(ct_img.shape, dtype=np.uint16)

    mask_files = sorted([p for p in seg_dir.glob("*.nii*") if p.is_file()])
    for idx, mpath in enumerate(mask_files, start=1):
        m = nib.load(str(mpath)).get_fdata()
        label[m > 0.5] = idx

    nib.save(nib.Nifti1Image(label, ct_img.affine), str(out_path))
    return out_path


