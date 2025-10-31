"""
2D slice visualization utilities for 3D medical segmentation.

Provides functions to visualize 3D segmentation results as 2D slices,
overlays, and GIF animations across axial, coronal, and sagittal planes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union, Dict, List
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
try:
    from PIL import Image
except ImportError:
    try:
        import Image
    except ImportError:
        raise ImportError("PIL/Pillow is required for GIF generation. Install with: pip install pillow")
import torch


# Color maps for segmentation overlays
BRATS_COLORS = {
    0: [0, 0, 0, 0],      # Background: transparent
    1: [255, 0, 0, 180],  # NCR/NET: red with transparency
    2: [0, 255, 0, 180],  # ED: green with transparency
    3: [0, 0, 255, 180],  # ET: blue with transparency
}

MSD_LIVER_COLORS = {
    0: [0, 0, 0, 0],      # Background: transparent
    1: [0, 255, 0, 180],  # Liver: green with transparency
    2: [255, 0, 0, 180],  # Tumor: red with transparency
}

DEFAULT_COLORS = {
    0: [0, 0, 0, 0],      # Background: transparent
    1: [255, 0, 0, 180],  # Class 1: red
    2: [0, 255, 0, 180],  # Class 2: green
    3: [0, 0, 255, 180],  # Class 3: blue
}


def get_colormap(dataset_name: str, num_classes: int) -> Dict[int, List[int]]:
    """Get color map for a dataset."""
    name = dataset_name.lower()
    if name == "brats":
        return BRATS_COLORS
    elif name == "msd_liver":
        return MSD_LIVER_COLORS
    else:
        # Default: extend colors for multi-class
        colors = dict(DEFAULT_COLORS)
        if num_classes > 4:
            # Generate additional colors for more classes
            for i in range(4, num_classes):
                hue = (i * 137.508) % 360  # Golden angle for color distribution
                rgb = matplotlib.colors.hsv_to_rgb([hue / 360, 0.8, 0.9])
                colors[i] = [int(c * 255) for c in rgb] + [180]
        return colors


def extract_slice(
    volume: np.ndarray,
    slice_index: int,
    plane: str = "axial"
) -> np.ndarray:
    """
    Extract a 2D slice from a 3D volume.
    
    Args:
        volume: 3D array of shape [D, H, W] or [C, D, H, W]
        slice_index: Index of the slice to extract
        plane: "axial", "coronal", or "sagittal"
    
    Returns:
        2D array slice
    """
    if len(volume.shape) == 4:
        # Multi-channel: take first channel or average
        volume = volume[0] if volume.shape[0] == 1 else volume.mean(axis=0)
    
    if plane == "axial":
        # Axial: slice along depth (first dimension)
        if slice_index >= volume.shape[0]:
            slice_index = volume.shape[0] - 1
        return volume[slice_index, :, :]
    elif plane == "coronal":
        # Coronal: slice along height (second dimension)
        if slice_index >= volume.shape[1]:
            slice_index = volume.shape[1] - 1
        return volume[:, slice_index, :]
    elif plane == "sagittal":
        # Sagittal: slice along width (third dimension)
        if slice_index >= volume.shape[2]:
            slice_index = volume.shape[2] - 1
        return volume[:, :, slice_index]
    else:
        raise ValueError(f"Unknown plane: {plane}. Must be 'axial', 'coronal', or 'sagittal'")


def create_overlay(
    image_slice: np.ndarray,
    mask_slice: np.ndarray,
    colormap: Dict[int, List[int]],
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create an overlay of segmentation mask on image.
    
    Args:
        image_slice: 2D grayscale image [H, W] in range [0, 1]
        mask_slice: 2D label mask [H, W] with integer class labels
        colormap: Dictionary mapping class labels to RGBA colors
        alpha: Transparency factor for overlay
    
    Returns:
        RGB image array [H, W, 3] with overlay
    """
    # Normalize image to [0, 1] if needed
    if image_slice.max() > 1.0:
        image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min() + 1e-8)
    
    # Convert to RGB
    rgb_image = np.stack([image_slice] * 3, axis=-1)
    
    # Create colored mask
    overlay = rgb_image.copy()
    mask_colored = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
    
    for label, color in colormap.items():
        mask = (mask_slice == label)
        if mask.any():
            mask_colored[mask] = np.array(color, dtype=np.float32) / 255.0
    
    # Blend overlay
    alpha_mask = mask_colored[:, :, 3:4] * alpha
    overlay = overlay * (1 - alpha_mask) + mask_colored[:, :, :3] * alpha_mask
    
    return np.clip(overlay, 0, 1)


def save_slice_triptych(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: Optional[np.ndarray],
    slice_index: int,
    plane: str,
    out_path: Path,
    dataset_name: str = "brats",
    num_classes: int = 4,
    dpi: int = 150
) -> None:
    """
    Save a triptych visualization: image, GT overlay, prediction overlay.
    
    Args:
        image: 3D image array [C, D, H, W] or [D, H, W]
        ground_truth: 3D label array [D, H, W] or [1, D, H, W]
        prediction: Optional 3D prediction array [D, H, W] or [C, D, H, W]
        slice_index: Slice index to visualize
        plane: "axial", "coronal", or "sagittal"
        out_path: Output file path
        dataset_name: Dataset name for colormap
        num_classes: Number of classes
        dpi: DPI for saved figure
    """
    # Convert torch tensors to numpy
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()
    if prediction is not None and isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    
    # Handle channel dimension
    if len(image.shape) == 4:
        # Multi-channel: use first channel or average for display
        if image.shape[0] > 1:
            image_display = image.mean(axis=0)
        else:
            image_display = image[0]
    else:
        image_display = image
    
    # Handle label dimension
    if len(ground_truth.shape) == 4:
        ground_truth = ground_truth[0] if ground_truth.shape[0] == 1 else ground_truth.squeeze()
    if prediction is not None:
        if len(prediction.shape) == 4:
            # If multi-channel (logits), take argmax
            if prediction.shape[0] > 1:
                prediction = np.argmax(prediction, axis=0)
            else:
                prediction = prediction[0]
        elif len(prediction.shape) == 3:
            prediction = prediction
    
    # Extract slices
    img_slice = extract_slice(image_display, slice_index, plane)
    gt_slice = extract_slice(ground_truth, slice_index, plane).astype(np.int32)
    pred_slice = extract_slice(prediction, slice_index, plane).astype(np.int32) if prediction is not None else None
    
    # Get colormap
    colormap = get_colormap(dataset_name, num_classes)
    
    # Create figure
    fig, axes = plt.subplots(1, 3 if prediction is not None else 2, figsize=(15, 5))
    if prediction is None:
        axes = [axes, plt.subplot(1, 2, 2)]
    
    # Image
    axes[0].imshow(img_slice, cmap="gray")
    axes[0].set_title(f"Input Image ({plane}, slice {slice_index})")
    axes[0].axis("off")
    
    # Ground truth overlay
    gt_overlay = create_overlay(img_slice, gt_slice, colormap, alpha=0.6)
    axes[1].imshow(gt_overlay)
    axes[1].set_title(f"Ground Truth ({plane}, slice {slice_index})")
    axes[1].axis("off")
    
    # Prediction overlay
    if prediction is not None:
        pred_overlay = create_overlay(img_slice, pred_slice, colormap, alpha=0.6)
        axes[2].imshow(pred_overlay)
        axes[2].set_title(f"Prediction ({plane}, slice {slice_index})")
        axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def export_gif_over_slices(
    image: np.ndarray,
    mask: np.ndarray,
    plane: str,
    out_path: Path,
    dataset_name: str = "brats",
    num_classes: int = 4,
    step: int = 1,
    alpha: float = 0.6,
    duration: int = 100
) -> None:
    """
    Export a GIF animation showing slices across a plane with overlay.
    
    Args:
        image: 3D image array [C, D, H, W] or [D, H, W]
        mask: 3D label array [D, H, W] or [1, D, H, W]
        plane: "axial", "coronal", or "sagittal"
        out_path: Output GIF file path
        dataset_name: Dataset name for colormap
        num_classes: Number of classes
        step: Step size between slices
        alpha: Transparency for overlay
        duration: Frame duration in milliseconds
    """
    # Convert torch tensors to numpy
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Handle dimensions
    if len(image.shape) == 4:
        if image.shape[0] > 1:
            image = image.mean(axis=0)
        else:
            image = image[0]
    
    if len(mask.shape) == 4:
        mask = mask[0] if mask.shape[0] == 1 else mask.squeeze()
    
    # Determine slice range
    if plane == "axial":
        num_slices = image.shape[0]
    elif plane == "coronal":
        num_slices = image.shape[1]
    elif plane == "sagittal":
        num_slices = image.shape[2]
    else:
        raise ValueError(f"Unknown plane: {plane}")
    
    # Get colormap
    colormap = get_colormap(dataset_name, num_classes)
    
    # Collect frames
    frames = []
    for i in range(0, num_slices, step):
        img_slice = extract_slice(image, i, plane)
        mask_slice = extract_slice(mask, i, plane).astype(np.int32)
        
        overlay = create_overlay(img_slice, mask_slice, colormap, alpha=alpha)
        
        # Convert to PIL Image
        overlay_uint8 = (overlay * 255).astype(np.uint8)
        frame = Image.fromarray(overlay_uint8)
        frames.append(frame)
    
    # Save as GIF
    if frames:
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )


def visualize_predictions(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: Optional[torch.Tensor],
    output_dir: Path,
    case_id: str,
    dataset_name: str = "brats",
    num_classes: int = 4,
    num_slices: int = 5,
    planes: Tuple[str, ...] = ("axial",)
) -> None:
    """
    Generate comprehensive visualizations for a batch of predictions.
    
    Args:
        images: Batch of images [B, C, D, H, W]
        labels: Batch of labels [B, 1, D, H, W] or [B, C, D, H, W]
        predictions: Batch of predictions [B, C, D, H, W] or [B, 1, D, H, W]
        output_dir: Directory to save visualizations
        case_id: Case identifier
        dataset_name: Dataset name
        num_classes: Number of classes
        num_slices: Number of slices to visualize per plane
        planes: Tuple of planes to visualize
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each item in batch
    batch_size = images.shape[0]
    for b in range(batch_size):
        img = images[b]
        lbl = labels[b]
        pred = predictions[b] if predictions is not None else None
        
        case_output_dir = output_dir / f"{case_id}_batch{b}" if batch_size > 1 else output_dir / case_id
        case_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle predictions: if logits, convert to class indices
        if pred is not None:
            if len(pred.shape) == 5 and pred.shape[1] > 1:
                # Multi-channel logits: take argmax
                pred = torch.argmax(pred, dim=1, keepdim=True)
        
        # Handle labels: if one-hot, convert to indices
        if len(lbl.shape) == 5 and lbl.shape[1] > 1:
            lbl = torch.argmax(lbl, dim=1, keepdim=True)
        
        # Convert to numpy
        img_np = img.detach().cpu().numpy()
        lbl_np = lbl.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy() if pred is not None else None
        
        # Determine slice indices (equally spaced)
        for plane in planes:
            if plane == "axial":
                num_slices_available = img_np.shape[-3] if len(img_np.shape) == 4 else img_np.shape[1]
            elif plane == "coronal":
                num_slices_available = img_np.shape[-2] if len(img_np.shape) == 4 else img_np.shape[2]
            elif plane == "sagittal":
                num_slices_available = img_np.shape[-1] if len(img_np.shape) == 4 else img_np.shape[3]
            else:
                continue
            
            slice_indices = np.linspace(0, num_slices_available - 1, num_slices, dtype=int)
            
            # Save individual slices
            for slice_idx in slice_indices:
                out_file = case_output_dir / f"{plane}_slice{slice_idx:03d}.png"
                save_slice_triptych(
                    img_np, lbl_np, pred_np,
                    slice_idx, plane, out_file,
                    dataset_name, num_classes
                )
            
            # Export GIF for ground truth
            gt_gif_path = case_output_dir / f"{plane}_ground_truth.gif"
            export_gif_over_slices(
                img_np, lbl_np, plane, gt_gif_path,
                dataset_name, num_classes, step=max(1, num_slices_available // 50)
            )
            
            # Export GIF for prediction if available
            if pred_np is not None:
                pred_gif_path = case_output_dir / f"{plane}_prediction.gif"
                export_gif_over_slices(
                    img_np, pred_np, plane, pred_gif_path,
                    dataset_name, num_classes, step=max(1, num_slices_available // 50)
                )

