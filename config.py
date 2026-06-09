"""
Configuration for the improved dermoscopy classification pipeline.
All paths, hyperparameters, and constants in one place.
"""
from pathlib import Path

# ============================================================
# Paths
# ============================================================
DATA_ROOT     = Path("/data/home/ha241003/Maiques-Lab/dermoscopy")
PANDERM_REPO  = Path("/data/home/ha241003/Maiques-Lab/PanDerm")   # git clone https://github.com/SiyuanYan1/PanDerm.git
PANDERM_CLASS = PANDERM_REPO / "classification"

CHECKPOINT_LARGE = Path("/data/home/ha241003/Maiques-Lab/panderm_ll_data6_checkpoint-499.pth")
CHECKPOINT_BASE  = Path("/data/home/ha241003/Maiques-Lab/panderm_bb_data6_checkpoint-499.pth")

PIPELINE_DIR  = Path("/data/home/ha241003/Maiques-Lab/improved_pipeline")
OUTPUT_DIR    = PIPELINE_DIR / "results"
SEGMENTED_DIR = PIPELINE_DIR / "segmented_cache"
FEATURES_DIR  = PIPELINE_DIR / "features"
CSV_DIR       = PIPELINE_DIR / "cross-fold-csv"
CLINICAL_DIR  = Path("/data/home/ha241003/Maiques-Lab/clinical_outputs")

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

# ============================================================
# Segmentation
# ============================================================
USE_SEGMENTATION  = True
MORPH_KERNEL_SIZE = 15    # kernel for morphological closing
MIN_LESION_RATIO  = 0.01  # fallback to full image if mask < 1% of area
CROP_MARGIN       = 0.1   # 10% margin around lesion bounding box
