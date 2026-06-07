"""
Step 2: Lesion segmentation preprocessing.

Segments the lesion from the background using a simple color-based approach,
crops to the lesion bounding box, and saves processed images.

This module is designed to be modular: swap ColorBasedSegmenter for a
deep-learning segmenter (e.g., PanDerm segmentation, SAM) by subclassing
LesionSegmenter.

Usage:
    python segment_lesions.py
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy import ndimage
from skimage.measure import label as sk_label
import matplotlib.pyplot as plt
from config import (
    OUTPUT_DIR, SEGMENTED_DIR, IMAGE_SIZE,
    MORPH_KERNEL_SIZE, MIN_LESION_RATIO, CROP_MARGIN,
)


class LesionSegmenter:
    """Base class for lesion segmentation. Subclass to implement custom segmenters."""

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment the lesion from the background.

        Args:
            image: RGB image as numpy array (H, W, 3), uint8.

        Returns:
            Binary mask (H, W) where 1 = lesion, 0 = background.
        """
        raise NotImplementedError


class ColorBasedSegmenter(LesionSegmenter):
    """
    Simple color-based segmentation using LAB color space + Otsu thresholding.

    Steps:
      1. Convert to LAB color space
      2. Gaussian blur on L channel
      3. Otsu threshold (lesions are typically darker than surrounding skin)
      4. Morphological closing to fill small gaps
      5. Fill holes
      6. Keep largest connected component
      7. Fallback to full image if mask is too small
    """

    def __init__(self, morph_kernel_size: int = 15, min_lesion_ratio: float = 0.01):
        self.morph_kernel_size = morph_kernel_size
        self.min_lesion_ratio = min_lesion_ratio

    def segment(self, image: np.ndarray) -> np.ndarray:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB) 
        l_channel = lab[:, :, 0]

        # Gaussian blur to reduce noise - smooths the image - removes the tiny hairs or individual skin pores
        blurred = cv2.GaussianBlur(l_channel, (11, 11), 0)

        # Otsu threshold (inverted: dark lesion → foreground) - best cutoff value to separate dark pixels from light pixels
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological closing - if there are small gaps or holes in the dark lesion it fills them for solid blob 
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size)
        )
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Fill holes
        filled = ndimage.binary_fill_holes(closed).astype(np.uint8)

        # Keep largest connected component - identifies the largest connected component, separates from hair or pores.
        mask = self._largest_connected_component(filled)

        # Fallback: if mask is too small, use full image
        total_area = mask.shape[0] * mask.shape[1]
        if mask.sum() < self.min_lesion_ratio * total_area:
            mask = np.ones_like(mask, dtype=np.uint8)

        return mask

    @staticmethod
    def _largest_connected_component(binary_img: np.ndarray) -> np.ndarray:
        """
        Keep only the largest connected component.
        Replicates logic from PanDerm/segmentation/utils/train_utils.py.
        """
        labeled = sk_label(binary_img, background=0)
        if labeled.max() == 0:
            return binary_img

        largest_label = 0
        largest_size = 0
        for region_label in range(1, labeled.max() + 1):
            size = (labeled == region_label).sum()
            if size > largest_size:
                largest_size = size
                largest_label = region_label

        lcc = (labeled == largest_label).astype(np.uint8)
        lcc = ndimage.binary_fill_holes(lcc).astype(np.uint8)
        return lcc


def apply_mask_and_crop(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: int = 224,
    margin: float = 0.1,
) -> np.ndarray:
    """
    Crop image to the lesion bounding box with margin, apply mask, resize.

    Args:
        image: RGB image (H, W, 3).
        mask: Binary mask (H, W).
        target_size: Output size.
        margin: Fractional margin around bounding box.

    Returns:
        Cropped and resized RGB image (target_size, target_size, 3).
    """
    h, w = mask.shape # height, width

    # Find bounding box - the smallest rectangle that fits around the lesion
    ys, xs = np.where(mask > 0) #identifies the coordinates of the lesion (W pixel)
    if len(ys) == 0: 
        # No lesion found, return resized original
        return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    # Add margin
    box_h = y_max - y_min
    box_w = x_max - x_min
    pad_y = int(box_h * margin)
    pad_x = int(box_w * margin)

    y_min = max(0, y_min - pad_y)
    y_max = min(h, y_max + pad_y)
    x_min = max(0, x_min - pad_x)
    x_max = min(w, x_max + pad_x)

    # Crop
    cropped = image[y_min:y_max, x_min:x_max].copy()

    # Apply mask within crop (set background to black)
    crop_mask = mask[y_min:y_max, x_min:x_max]
    cropped[crop_mask == 0] = 0

    # Resize
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    return resized


def process_dataset(
    manifest_csv: Path,
    output_dir: Path,
    segmenter: LesionSegmenter = None,
    target_size: int = 224,
    margin: float = 0.1,
) -> pd.DataFrame:
    """
    Process all images: segment, crop, save.

    Returns updated DataFrame with 'segmented_path' column.
    """
    if segmenter is None:
        segmenter = ColorBasedSegmenter()

    df = pd.read_csv(manifest_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    segmented_paths = []
    n_total = len(df)
    n_fallback = 0

    for idx, row in df.iterrows():
        img_path = Path(row["image_path"])
        diagnosis = row["diagnosis"]
        folder_name = row.get("folder_name", row["patient_id"])

        # Output path mirrors original structure
        out_dir = output_dir / diagnosis / str(folder_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / img_path.name

        if out_path.exists():
            segmented_paths.append(str(out_path))
            if (idx + 1) % 100 == 0:
                print(f"  [{idx + 1}/{n_total}] (cached)")
            continue

        # Load image
        try:
            pil_img = Image.open(str(img_path)).convert("RGB")
            image = np.array(pil_img)
        except Exception as e:
            print(f"  ERROR loading {img_path}: {e}")
            segmented_paths.append(str(img_path))  # fallback to original
            continue

        # Segment
        mask = segmenter.segment(image)

        # Check if fallback (full image)
        if mask.sum() == mask.shape[0] * mask.shape[1]:
            n_fallback += 1

        # Crop and save
        cropped = apply_mask_and_crop(image, mask, target_size, margin)
        Image.fromarray(cropped).save(str(out_path), quality=95)

        segmented_paths.append(str(out_path))

        if (idx + 1) % 50 == 0:
            print(f"  [{idx + 1}/{n_total}] processed")

    df["segmented_path"] = segmented_paths
    print(f"\nSegmentation complete: {n_total} images, {n_fallback} fallbacks to full image")
    return df


def generate_qc_montage(
    manifest_df: pd.DataFrame,
    segmenter: LesionSegmenter,
    output_path: Path,
    n_samples: int = 20,
    seed: int = 42,
):
    """
    Generate a visual QC montage showing: original | mask overlay | cropped
    for a random sample of images.
    """
    rng = np.random.RandomState(seed)
    sample_idx = rng.choice(len(manifest_df), size=min(n_samples, len(manifest_df)), replace=False)
    sample_idx.sort()

    n = len(sample_idx)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, idx in enumerate(sample_idx):
        row = manifest_df.iloc[idx]
        img_path = row["image_path"]

        try:
            pil_img = Image.open(img_path).convert("RGB")
            image = np.array(pil_img)
        except Exception:
            continue

        mask = segmenter.segment(image)
        cropped = apply_mask_and_crop(image, mask, 224, CROP_MARGIN)

        # Original
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"{row['diagnosis']}/{row['patient_id']}", fontsize=8)
        axes[i, 0].axis("off")

        # Mask overlay
        overlay = image.copy()
        overlay[mask == 1] = (overlay[mask == 1] * 0.6 + np.array([0, 255, 0]) * 0.4).astype(np.uint8)
        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title("Mask overlay", fontsize=8)
        axes[i, 1].axis("off")

        # Cropped
        axes[i, 2].imshow(cropped)
        axes[i, 2].set_title("Cropped lesion", fontsize=8)
        axes[i, 2].axis("off")

    plt.suptitle("Segmentation QC Montage", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"QC montage saved to: {output_path}")


def main():
    print("Step 2: Lesion segmentation preprocessing")

    manifest_csv = OUTPUT_DIR / "dataset_manifest.csv"
    if not manifest_csv.exists():
        print(f"ERROR: Manifest not found at {manifest_csv}")
        print("Run prepare_data.py first.")
        return

    segmenter = ColorBasedSegmenter(
        morph_kernel_size=MORPH_KERNEL_SIZE,
        min_lesion_ratio=MIN_LESION_RATIO,
    )

    # Process dataset
    print(f"\nProcessing images (saving to {SEGMENTED_DIR})...")
    df = process_dataset(manifest_csv, SEGMENTED_DIR, segmenter, IMAGE_SIZE, CROP_MARGIN)

    # Save updated manifest
    updated_csv = OUTPUT_DIR / "dataset_manifest.csv"
    df.to_csv(updated_csv, index=False)
    print(f"Updated manifest saved to: {updated_csv}")

    # Generate QC montage
    qc_path = OUTPUT_DIR / "segmentation_qc_montage.png"
    print(f"\nGenerating QC montage...")
    generate_qc_montage(df, segmenter, qc_path)


if __name__ == "__main__":
    main()
