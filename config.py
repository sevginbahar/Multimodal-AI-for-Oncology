"""
Configuration for the improved dermoscopy classification pipeline.
All paths, hyperparameters, and constants in one place.
"""
from pathlib import Path

# ============================================================
# Segmentation backend — defined first because FEATURES_DIR and
# SEGMENTED_DIR below are both keyed by it, so switching this ONE line
# automatically gives every downstream stage (segment, extract_features,
# evaluate) its own separate output location. Nothing from one backend can
# silently overwrite another's manifest / features / results.
#   "none"  -> no segmentation; segmented_path = image_path (the raw PanDerm-frozen arm)
#   "color"    -> LAB + Otsu + morphological closing (ColorBasedSegmenter)
#   "sam"      -> SAM2, centre-point prompt (needs `pip install` — see segment_lesions.py)
#   "color_v2" -> Otsu inside the dermoscope field of view (vignette/rim removed),
#                 on a DullRazor hair-removed copy, centre-weighted component
#                 (ColorBasedSegmenterV2)
#   "sam_v2"   -> SAM2 prompted with box + positive/negative points from the
#                 color_v2 estimate, mask chosen by plausibility, on the
#                 hair-removed copy (PromptedSAMSegmenter)
#   "sam_v1_pre" -> ablation: v2 preprocessing (1024 px, DullRazor, field of view)
#                 + SAM v1 prompt (one image-centre point, highest score), mask
#                 clipped to the field of view (SAMV1PreprocessedSegmenter)
#   v2 crops keep the original pixels; hair removal is used only to find the mask.
# ============================================================

# ============================================================
# Paths
# ------------------------------------------------------------
# To switch cohorts: comment out the active block, uncomment the other.
# Each cohort has its own OUTPUT_DIR so one run can never overwrite the
# other's manifest / features / results.
# ============================================================


# ============================================================
# Segmentation (SEGMENTATION_BACKEND itself is set at the top of this file)
# ============================================================
USE_SEGMENTATION  = True
MORPH_KERNEL_SIZE = 15    # kernel for morphological closing
MIN_LESION_RATIO  = 0.01  # fallback to full image if mask < 1% of area
CROP_MARGIN       = 0.1   # 10% margin around lesion bounding box
SEGMENTATION_BACKEND = "sam_v1_pre"  # "none", "color", "sam", "color_v2", "sam_v2", "sam_v1_pre" (see top of file)

DATA_ROOT  = Path("C:\\Users\\qp263329\\OneDrive - Queen Mary, University of London\\Desktop\\PhD_Data\\dermoscopy")
OUTPUT_DIR = Path("C:\\Users\\qp263329\\OneDrive - Queen Mary, University of London\\Desktop\\PhD_Data\\inhouse_pipeline\\outputs")

SEGMENTED_DIR = OUTPUT_DIR / "segmented_cache"
FEATURES_DIR  = OUTPUT_DIR / "features" / SEGMENTATION_BACKEND
EVAL_DIR      = OUTPUT_DIR / "results" / SEGMENTATION_BACKEND
CSV_DIR       = OUTPUT_DIR / "cross-fold-csv"

METADATA_CSV  = None  # set to a CSV path for flat-folder datasets (e.g. MILK10K)
PANDERM_REPO  = Path("C:\\Users\\qp263329\\PanDerm")   # git clone https://github.com/SiyuanYan1/PanDerm.git
PANDERM_CLASS = PANDERM_REPO / "classification"

CHECKPOINT_LARGE = Path("C:\\Users\\qp263329\\OneDrive - Queen Mary, University of London\\Desktop\\PhD_Data\\panderm_ll_data6_checkpoint-499.pth")
CHECKPOINT_BASE  = Path("C:\\Users\\qp263329\\OneDrive - Queen Mary, University of London\\Desktop\\PhD_Data\\panderm_bb_data6_checkpoint-499.pth")

PIPELINE_DIR  = Path("C:\\Users\\qp263329\\OneDrive - Queen Mary, University of London\\Desktop\\Multimodal-AI-for-Oncology")
CLINICAL_DIR   = Path("C:\\Users\\qp263329\\OneDrive - Queen Mary, University of London\\Desktop\\PhD_Data\\clinical_outputs")
CLINICAL_INPUT = Path("C:\\Users\\qp263329\\OneDrive - Queen Mary, University of London\\Desktop\\PhD_Data\\harmonized_pathology_reports.csv")  # raw clinical data

print(f"[config] DATA_ROOT={DATA_ROOT}")
print(f"[config] OUTPUT_DIR={OUTPUT_DIR}  FEATURES_DIR={FEATURES_DIR}")


def input_image(image_path) -> str:
    """The image the model should see for a manifest row, under the CURRENT
    SEGMENTATION_BACKEND: the raw image for "none", otherwise its crop in
    segmented_cache/<backend>/, which mirrors DATA_ROOT's <class>/<patient>/<file>.

    Derived from image_path rather than read from the manifest's segmented_path
    column, because every segment run overwrites that column — it holds
    whichever backend was segmented last, not the one configured now.
    """
    if SEGMENTATION_BACKEND == "none":
        return str(image_path)
    return str(SEGMENTED_DIR / SEGMENTATION_BACKEND / Path(image_path).relative_to(DATA_ROOT))

# ============================================================
# Class definitions
# ============================================================
CLASS_NAMES = ["DN", "MIA", "Minsitu"]
CLASS_LABELS = {"DN": 0, "MIA": 1, "Minsitu": 2}
DISPLAY_NAMES = {
    "DN":      "Dysplastic Nevus",
    "MIA":     "Melanoma Stage IA",
    "Minsitu": "Melanoma In Situ",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ============================================================
# PanDerm model configuration
# ============================================================
MODEL_VARIANT = "large"  # "large" (ViT-L/16, 1024-dim) or "base" (ViT-B/16, 768-dim)
EMBED_DIM     = {"large": 1024, "base": 768}
IMAGE_SIZE    = 224
NB_CLASSES    = 3

# Official normalisation from PanDerm builder.py
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.228, 0.224, 0.225]  # NOTE: builder.py uses 0.228, NOT 0.229

BATCH_SIZE  = 32
NUM_WORKERS = 4  # set to 0 if DataLoader hangs

# ============================================================
# Cross-validation
# ============================================================
N_FOLDS     = 5
RANDOM_SEED = 42

# ============================================================
# Classifier (logistic regression)
# ============================================================
LOGREG_C        = 1.0
LOGREG_MAX_ITER = 2000


