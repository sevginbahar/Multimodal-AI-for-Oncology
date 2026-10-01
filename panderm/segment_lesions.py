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
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy import ndimage
from skimage.measure import label as sk_label
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
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


class SAMSegmenter(LesionSegmenter):
    """
    SAM2, prompted with a centre point — dermoscopy lesions are
    conventionally centred in the frame.

    Requires the OFFICIAL Meta package :
        pip install "git+https://github.com/facebookresearch/sam2.git"

    Checkpoint auto-downloads from HuggingFace Hub on first use — no manual
    .pth management needed. Uses multimask_output + picks SAM's own
    highest-confidence mask, since a single point prompt is ambiguous
    (could mean "the lesion", "the whole framed circle", etc.).

    UNTESTED against a real sam2 install as of writing — the exact
    `SAM2ImagePredictor` API can drift between package versions.
    """

    def __init__(self, model_id: str = "facebook/sam2.1-hiera-small"):
        try:
            import torch
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            raise NotImplementedError(
                "sam2 package not installed. Run:\n"
                '  pip install "git+https://github.com/facebookresearch/sam2.git"\n'
                f"(import error: {e})"
            )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = SAM2ImagePredictor.from_pretrained(model_id)
        self.predictor.model.to(self.device)

    def segment(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        self.predictor.set_image(image)
        point = np.array([[w // 2, h // 2]])   # centre-point prompt
        label = np.array([1])                  # foreground
        masks, scores, _ = self.predictor.predict(
            point_coords=point, point_labels=label, multimask_output=True
        )
        best = masks[int(np.argmax(scores))]
        return best.astype(np.uint8)


class IdentitySegmenter(LesionSegmenter):
    """
    The "none" arm of the segmentation ablation — keeps the full frame,
    no cropping. Included for interface symmetry; process_dataset() actually
    fast-paths "none" (segmented_path = image_path, no image processing at
    all) rather than routing through this + apply_mask_and_crop, to exactly
    reproduce the prior no-segmentation runs (avoids a redundant resize).
    """

    def segment(self, image: np.ndarray) -> np.ndarray:
        return np.ones(image.shape[:2], dtype=np.uint8)


def build_segmenter(backend: str) -> "LesionSegmenter | None":
    """
    Factory keyed off config.SEGMENTATION_BACKEND. Returns None for "none"
    (process_dataset fast-paths that case and never calls .segment()).
    """
    from config import MORPH_KERNEL_SIZE, MIN_LESION_RATIO

    backend = backend.lower()
    if backend == "none":
        return None
    if backend == "color":
        return ColorBasedSegmenter(MORPH_KERNEL_SIZE, MIN_LESION_RATIO)
    if backend == "sam":
        return SAMSegmenter()
    if backend == "color_v2":
        return ColorBasedSegmenterV2()
    if backend == "sam_v2":
        return PromptedSAMSegmenter()
    if backend == "sam_v1_pre":
        return SAMV1PreprocessedSegmenter()
    raise ValueError(f"Unknown SEGMENTATION_BACKEND: {backend!r} "
                     f"(expected 'none', 'color', 'sam', 'color_v2', 'sam_v2' or 'sam_v1_pre')")


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


# ── v2 segmenters: field-of-view masking + DullRazor + prompted SAM ─────────
# Fix the failure modes seen with "color"/"sam" on the in-house cohort:
#   - Otsu picked the dark vignette / scope rim / corners as "lesion"
#   - dark hairs got joined onto the lesion by morphological closing
#   - SAM's single centre point was ambiguous (whole / part / sub-part) and
#     argmax(score) favoured tiny sub-part masks
# Masks are computed at WORK_SIZE (long side) on a hair-removed copy, then
# upsampled to full resolution; the crop itself keeps the ORIGINAL pixels,
# since inpainting also smooths pigment network / dots inside the lesion.

WORK_SIZE = 1024


def _odd(n: float) -> int:
    n = max(3, int(round(n)))
    return n if n % 2 else n + 1


def to_work_res(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    scale = WORK_SIZE / max(h, w)
    if scale >= 1:
        return image
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def dullrazor(rgb: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """DullRazor-style hair removal (Lee et al., 1997): black-hat with a cross
    kernel picks thin dark structures, threshold -> hair mask, then inpaint.
    Kernel scales with image size (17 px at ~1000 px, as in common
    implementations). Returns (hair-free RGB, hair mask)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    k = _odd(0.017 * max(gray.shape))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                                cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k)))
    _, hair = cv2.threshold(cv2.GaussianBlur(blackhat, (3, 3), 0), 10, 255, cv2.THRESH_BINARY)
    hair = cv2.dilate(hair, np.ones((3, 3), np.uint8))
    return cv2.inpaint(rgb, hair, 5, cv2.INPAINT_TELEA), hair > 0


def field_of_view_mask(rgb: np.ndarray) -> np.ndarray:
    """Usable dermoscope field: excludes dark vignette corners / scope rim, i.e.
    dark regions connected to the image border. The remaining field is eroded
    by ~2% to drop the soft dark gradient at the rim. A dark lesion that
    touches the frame is not removed because 'dark' is relative to the frame
    centre and far darker than pigmented skin (<35% of the centre median)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    ref = float(np.median(gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4]))
    dark = (gray < max(20.0, 0.35 * ref)).astype(np.uint8)
    n, lab = cv2.connectedComponents(dark)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    outside = np.isin(lab, border[border > 0])
    fov = ~outside
    if outside.mean() > 0.01:   # there is a vignette -> strip the rim gradient too
        k = _odd(0.02 * max(h, w))
        fov = cv2.erode(fov.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    return fov if fov.mean() > 0.2 else np.ones_like(fov)


def otsu_in_fov(rgb: np.ndarray, fov: np.ndarray,
                morph_frac: float = 0.012, min_ratio: float = 0.01) -> "np.ndarray | None":
    """Otsu on the L channel, threshold computed from field-of-view pixels only,
    lesion = connected component with the largest area weighted by closeness
    to the field-of-view centre (not simply the largest). None if nothing
    plausible (< min_ratio of the field)."""
    L = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 0], (_odd(0.01 * max(rgb.shape)),) * 2, 0)
    t, _ = cv2.threshold(L[fov].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = ((L <= t) & fov).astype(np.uint8)
    k = _odd(morph_frac * max(rgb.shape))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    binary = ndimage.binary_fill_holes(binary) & fov

    n, lab, stats, cent = cv2.connectedComponentsWithStats(binary.astype(np.uint8))
    if n <= 1:
        return None
    ys, xs = np.nonzero(fov)
    cy, cx = ys.mean(), xs.mean()
    radius = np.sqrt(fov.sum() / np.pi)
    area = stats[1:, cv2.CC_STAT_AREA].astype(float)
    dist = np.hypot(cent[1:, 0] - cx, cent[1:, 1] - cy) / radius
    # Radial position of every pixel; a component lying mostly in the outer
    # ring (r > 0.8) is a rim / corner / edge shadow remnant, not a lesion.
    gy, gx = np.indices(fov.shape)
    outer = np.hypot(gx - cx, gy - cy) / radius > 0.8
    for c in 1 + np.argsort(-(area * np.exp(-dist ** 2))):
        comp = lab == c
        if comp.sum() < min_ratio * fov.sum():
            break   # remaining candidates are smaller still
        if (comp & outer).sum() / comp.sum() <= 0.6:
            return comp
    return None


def _upsample(mask: np.ndarray, shape) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


class ColorBasedSegmenterV2(LesionSegmenter):
    """Otsu inside the dermoscope field of view, on a DullRazor hair-removed
    image, with centre-weighted component selection. Falls back to the field
    of view (not the raw frame) when no plausible lesion is found.
    `last_info` holds per-image QC values for process_dataset()."""

    def __init__(self):
        self.last_info = {}

    def segment(self, image: np.ndarray) -> np.ndarray:
        work = to_work_res(image)
        clean, hair = dullrazor(work)
        fov = field_of_view_mask(clean)
        mask = otsu_in_fov(clean, fov)
        self.last_info = {"hair_frac": float(hair.mean()), "fov_frac": float(fov.mean()),
                          "fallback": mask is None}
        if mask is None:
            mask = fov
        self.last_info["mask_frac_of_fov"] = float(mask.sum() / fov.sum())
        return _upsample(mask, image.shape)


class SAMV1PreprocessedSegmenter(LesionSegmenter):
    """Ablation: SAM v1's prompt and selection on the v2 preprocessing.
    Same working scale, DullRazor hair removal and field-of-view detection as
    the v2 segmenters, then exactly SAM v1: ONE positive point at the image
    centre, 3 candidates, highest SAM score kept — no box, no negative points,
    no clean-up, no plausibility rules, no fallback. The mask is clipped to the
    field of view. Compared with "sam" it isolates the preprocessing; compared
    with "sam_v2" it isolates the Otsu-based prompt + selection.
    If the image centre falls outside the field of view, the point moves to the
    field's centre (recorded as point_moved)."""

    def __init__(self, model_id: str = "facebook/sam2.1-hiera-small"):
        self._sam = SAMSegmenter(model_id)   # reuse loading / install check
        self.last_info = {}

    def segment(self, image: np.ndarray) -> np.ndarray:
        work = to_work_res(image)
        clean, hair = dullrazor(work)
        fov = field_of_view_mask(clean)
        h, w = fov.shape
        px, py = w // 2, h // 2
        moved = not fov[py, px]
        if moved:
            ys, xs = np.nonzero(fov)
            py, px = int(ys.mean()), int(xs.mean())

        pred = self._sam.predictor
        pred.set_image(clean)
        masks, scores, _ = pred.predict(point_coords=np.array([[px, py]], dtype=np.float32),
                                        point_labels=np.array([1]), multimask_output=True)
        best = int(np.argmax(scores))
        mask = masks[best].astype(bool) & fov
        self.last_info = {"hair_frac": float(hair.mean()), "fov_frac": float(fov.mean()),
                          "point_moved": moved, "sam_score": float(scores[best]),
                          "mask_frac_of_frame": float(mask.mean()),
                          "mask_frac_of_fov": float(mask.sum() / fov.sum()),
                          "fallback": False}
        return _upsample(mask, image.shape)


class PromptedSAMSegmenter(LesionSegmenter):
    """SAM2 with an unambiguous prompt instead of a lone centre point:
      box      around the v2 Otsu lesion (+5%), or the central field of view
               if Otsu found nothing
      point +  deepest interior point of that lesion estimate
      points - skin just outside the box, inside the field of view
    Run on the hair-removed image. Of SAM's candidate masks, keep the
    highest-scoring PLAUSIBLE one: contains the positive point, covers
    MIN_AREA..MAX_AREA of the field of view, and lies inside the field.
    If none is plausible, fall back to the Otsu estimate (then the field)."""

    MIN_AREA, MAX_AREA = 0.02, 0.95

    def __init__(self, model_id: str = "facebook/sam2.1-hiera-small"):
        self._sam = SAMSegmenter(model_id)   # reuse loading / install check
        self.last_info = {}

    @staticmethod
    def _prompts(prior: "np.ndarray | None", fov: np.ndarray):
        h, w = fov.shape
        region = prior if prior is not None else fov
        ys, xs = np.nonzero(region)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        if prior is None:   # no lesion estimate: central 60% of the field
            cy, cx, hh, hw = (y0 + y1) / 2, (x0 + x1) / 2, 0.3 * (y1 - y0), 0.3 * (x1 - x0)
            y0, y1, x0, x1 = cy - hh, cy + hh, cx - hw, cx + hw
            pos = (int(cx), int(cy))
        else:
            dt = cv2.distanceTransform(prior.astype(np.uint8), cv2.DIST_L2, 5)
            py, px = np.unravel_index(np.argmax(dt), dt.shape)
            pos = (int(px), int(py))
        py_, px_ = 0.05 * (y1 - y0), 0.05 * (x1 - x0)
        box = np.array([max(0, x0 - px_), max(0, y0 - py_), min(w - 1, x1 + px_), min(h - 1, y1 + py_)])
        # negative points: midway between each box edge and the field edge, if that spot is in the field
        bx0, by0, bx1, by1 = box
        mx, my = (bx0 + bx1) / 2, (by0 + by1) / 2
        cand = [(bx0 / 2, my), ((bx1 + w) / 2, my), (mx, by0 / 2), (mx, (by1 + h) / 2)]
        neg = [(int(x), int(y)) for x, y in cand if 0 <= int(y) < h and 0 <= int(x) < w and fov[int(y), int(x)]]
        return box, pos, neg

    def segment(self, image: np.ndarray) -> np.ndarray:
        work = to_work_res(image)
        clean, hair = dullrazor(work)
        fov = field_of_view_mask(clean)
        prior = otsu_in_fov(clean, fov)
        box, pos, neg = self._prompts(prior, fov)

        pred = self._sam.predictor
        pred.set_image(clean)
        coords = np.array([pos] + neg, dtype=np.float32)
        labels = np.array([1] + [0] * len(neg))
        masks, scores, _ = pred.predict(point_coords=coords, point_labels=labels,
                                        box=box, multimask_output=True)
        fov_area = fov.sum()
        best, best_score = None, -1.0
        for m, s in zip(masks.astype(bool), scores):
            # SAM masks carry speckle islands / holes at the border: keep only the
            # component containing the positive point, then fill its holes
            if m[pos[1], pos[0]]:
                _, lab = cv2.connectedComponents(m.astype(np.uint8))
                m = ndimage.binary_fill_holes(lab == lab[pos[1], pos[0]])
            frac = (m & fov).sum() / fov_area
            if (m[pos[1], pos[0]] and self.MIN_AREA <= frac <= self.MAX_AREA
                    and (m & ~fov).sum() <= 0.05 * m.sum() and s > best_score):
                best, best_score = m & fov, float(s)

        source = "sam" if best is not None else ("otsu" if prior is not None else "fov")
        mask = best if best is not None else (prior if prior is not None else fov)
        self.last_info = {"hair_frac": float(hair.mean()), "fov_frac": float(fov.mean()),
                          "fallback": source != "sam", "mask_source": source,
                          "sam_score": best_score if best is not None else np.nan,
                          "mask_frac_of_fov": float(mask.sum() / fov_area)}
        return _upsample(mask, image.shape)


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
    segmenter: "LesionSegmenter | None" = None,
    target_size: int = 224,
    margin: float = 0.1,
    backend: str = "color",
) -> pd.DataFrame:
    """
    Process all images: segment, crop, save.

    backend="none" fast-paths to segmented_path == image_path — no image
    processing at all — to exactly reproduce a true no-segmentation run,
    rather than routing a full (all-ones) mask through the crop/resize
    machinery, which would apply a redundant extra resize on top of
    extract_features.py's own Resize/CenterCrop.

    Returns updated DataFrame with 'segmented_path' column.
    """
    df = pd.read_csv(manifest_csv)

    if backend == "none":
        df["segmented_path"] = df["image_path"]
        print(f"\nSegmentation backend = 'none': segmented_path = image_path "
              f"for all {len(df)} images (no image processing, no fallbacks).")
        return df

    if segmenter is None:
        segmenter = ColorBasedSegmenter()

    output_dir.mkdir(parents=True, exist_ok=True)

    segmented_paths = []
    qc_rows = []   # per-image diagnostics from segmenters that expose `last_info` (v2)
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

        # Check if fallback (full image, or the v2 segmenters' own fallback flag)
        info = getattr(segmenter, "last_info", None)
        if info:
            qc_rows.append({"image_path": str(img_path), "diagnosis": diagnosis,
                            "patient_id": row["patient_id"], **info})
            n_fallback += int(info.get("fallback", False))
        elif mask.sum() == mask.shape[0] * mask.shape[1]:
            n_fallback += 1

        # Crop and save
        cropped = apply_mask_and_crop(image, mask, target_size, margin)
        Image.fromarray(cropped).save(str(out_path), quality=95)

        segmented_paths.append(str(out_path))

        if (idx + 1) % 50 == 0:
            print(f"  [{idx + 1}/{n_total}] processed")

    df["segmented_path"] = segmented_paths
    print(f"\nSegmentation complete: {n_total} images, {n_fallback} fallbacks")
    if qc_rows:
        qc_path = output_dir / f"segmentation_qc_{backend}.csv"
        pd.DataFrame(qc_rows).to_csv(qc_path, index=False)
        print(f"Per-image QC ({len(qc_rows)} newly segmented images): {qc_path}")
    return df


def generate_qc_montage(
    manifest_df: pd.DataFrame,
    segmenter: "LesionSegmenter | None",
    output_path: Path,
    n_samples: int = 20,
    seed: int = 42,
):
    """
    Generate a visual QC montage showing: original | mask overlay | cropped
    for a random sample of images. No-op for backend="none" (segmenter=None)
    — nothing to visualize when nothing is being segmented.
    """
    if segmenter is None:
        print("  Segmenter is None (backend='none') — skipping QC montage.")
        return

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
    from config import SEGMENTATION_BACKEND

    manifest_csv = OUTPUT_DIR / "dataset_manifest.csv"
    if not manifest_csv.exists():
        print(f"ERROR: Manifest not found at {manifest_csv}")
        print("Run prepare_data.py first.")
        return

    # Same backend selection + per-backend output dir as run_pipeline.py's
    # stage_segment(), so running this file directly behaves identically.
    segmenter = build_segmenter(SEGMENTATION_BACKEND)
    backend_dir = SEGMENTED_DIR / SEGMENTATION_BACKEND

    print(f"Segmentation backend: {SEGMENTATION_BACKEND}")
    print(f"\nProcessing images (saving to {backend_dir})...")
    df = process_dataset(manifest_csv, backend_dir, segmenter, IMAGE_SIZE, CROP_MARGIN,
                         backend=SEGMENTATION_BACKEND)

    # Save updated manifest
    updated_csv = OUTPUT_DIR / "dataset_manifest.csv"
    df.to_csv(updated_csv, index=False)
    print(f"Updated manifest saved to: {updated_csv}")

    # Generate QC montage
    qc_path = OUTPUT_DIR / f"segmentation_qc_montage_{SEGMENTATION_BACKEND}.png"
    print(f"\nGenerating QC montage...")
    generate_qc_montage(df, segmenter, qc_path)


if __name__ == "__main__":
    main()
