"""
Multimodal AI for Oncology — End-to-End Pipeline Runner
========================================================
Runs all stages of the pipeline in order:

    1. prepare              — scan dataset, create manifest + k-fold splits
    2. segment              — colour-based lesion segmentation (LAB + Otsu)
    3. make_csv             — convert manifest to PanDerm fold CSVs
    4. finetune             — fine-tune PanDerm (one job per fold)
    5. extract_finetuned    — CLS feature extraction using fine-tuned checkpoints
    6. extract_pretrained   — CLS feature extraction using pretrained checkpoint only
    7. clinical_modality    — BioClinicalBERT embeddings from pathology reports
    8. fusion               — late fusion of image + clinical text features

Usage:
    # Full pipeline with fine-tuning
    python run_pipeline.py --stage all

    # Full pipeline without fine-tuning (pretrained features only)
    python run_pipeline.py --stage all --no-finetune

    # Individual stages
    python run_pipeline.py --stage prepare
    python run_pipeline.py --stage segment
    python run_pipeline.py --stage make_csv
    python run_pipeline.py --stage finetune
    python run_pipeline.py --stage extract_finetuned
    python run_pipeline.py --stage extract_pretrained
    python run_pipeline.py --stage clinical_modality
    python run_pipeline.py --stage fusion

    # Skip stages whose outputs already exist
    python run_pipeline.py --stage all --skip-existing
"""

import argparse
import sys
import time
import subprocess
from pathlib import Path

# ── Shared config ─────────────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_DIR))
from config import (
    DATA_ROOT, OUTPUT_DIR, SEGMENTED_DIR, CSV_DIR, CLINICAL_DIR,
    PANDERM_CLASS, CHECKPOINT_LARGE, CHECKPOINT_BASE,
    CLASS_NAMES, CLASS_LABELS, DISPLAY_NAMES,
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    MODEL_VARIANT, EMBED_DIM, NB_CLASSES,
    BATCH_SIZE, NUM_WORKERS, N_FOLDS, RANDOM_SEED,
    LOGREG_C, LOGREG_MAX_ITER,
    MORPH_KERNEL_SIZE, MIN_LESION_RATIO, CROP_MARGIN,
    SEGMENTATION_BACKEND,
)


# ── Argument parsing ──────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multimodal AI for Oncology — Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "prepare", "segment", "make_csv",
            "finetune", "extract_features", "evaluate",
            "clinical_modality",
            "fusion",
        ],
        default="all",
        help="Stage to run (default: all)",
    )
    parser.add_argument(
        "--no-finetune",
        action="store_true",
        help=(
            "When --stage all: skip fine-tuning and run extract_pretrained "
            "instead of extract_finetuned"
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a stage if its key output file already exists",
    )
    return parser.parse_args()


# ── Stage runner ──────────────────────────────────────────────────────────

def run_stage(name: str, func, skip_check=None, skip_existing: bool = False):
    if skip_existing and skip_check and skip_check():
        print(f"\n{'='*60}")
        print(f"SKIPPING  {name}  (outputs already exist)")
        print(f"{'='*60}")
        return

    print(f"\n{'='*60}")
    print(f"STAGE  {name}")
    print(f"{'='*60}")
    t0 = time.time()
    func()
    elapsed = time.time() - t0
    print(f"\n  Completed in {int(elapsed // 60)}m {elapsed % 60:.1f}s")


# ── Stages ────────────────────────────────────────────────────────────────

def stage_prepare():
    """Scan dataset → dataset_manifest.csv with 5-fold patient-level splits."""
    sys.path.insert(0, str(PIPELINE_DIR / "panderm"))
    from prepare_data import (
        scan_dataset, create_kfold_splits,
        validate_no_leakage, print_statistics,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Scanning: {DATA_ROOT}")
    df = scan_dataset(DATA_ROOT)
    print(f"Found {len(df)} images from {df['patient_id'].nunique()} patients")

    df = create_kfold_splits(df, N_FOLDS, RANDOM_SEED)

    if validate_no_leakage(df, N_FOLDS):
        print("No patient leakage detected across folds")

    out = OUTPUT_DIR / "dataset_manifest.csv"
    df.to_csv(out, index=False)
    print(f"Manifest saved: {out}")
    print_statistics(df, N_FOLDS)


def stage_segment():
    """Segment lesions — backend chosen by config.SEGMENTATION_BACKEND."""
    sys.path.insert(0, str(PIPELINE_DIR / "panderm"))
    from segment_lesions import build_segmenter, process_dataset, generate_qc_montage

    manifest_csv = OUTPUT_DIR / "dataset_manifest.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Run 'prepare' first — not found: {manifest_csv}")

    # Keyed by backend so switching SEGMENTATION_BACKEND never silently reuses
    # another backend's cached output at the same path.
    backend_dir = SEGMENTED_DIR / SEGMENTATION_BACKEND
    print(f"Segmentation backend: {SEGMENTATION_BACKEND}  ->  {backend_dir}")
    segmenter = build_segmenter(SEGMENTATION_BACKEND)
    df = process_dataset(
        manifest_csv, backend_dir, segmenter, IMAGE_SIZE, CROP_MARGIN,
        backend=SEGMENTATION_BACKEND,
    )
    df.to_csv(manifest_csv, index=False)
    print(f"Manifest updated with segmented_path: {manifest_csv}")
    generate_qc_montage(df, segmenter,
                        OUTPUT_DIR / f"segmentation_qc_montage_{SEGMENTATION_BACKEND}.png")


def stage_make_csv():
    """Convert manifest to one PanDerm-format CSV per fold (cross-fold-csv/<backend>/)."""
    script = PIPELINE_DIR / "panderm" / "make_panderm_csv.py"
    cmd = [sys.executable, str(script)]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


FINETUNE_DIR = OUTPUT_DIR / "checkpoints" / "finetune" / SEGMENTATION_BACKEND


def stage_finetune():
    """Partial fine-tune PanDerm per fold, then write each fold's test predictions."""
    script = PIPELINE_DIR / "panderm" / "panderm_finetuning.py"
    if not script.exists():
        raise FileNotFoundError(f"Fine-tuning script not found: {script}")
    cmd = [sys.executable, str(script)]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def stage_extract_features():
    """Extract frozen (pretrained) PanDerm CLS features and save to disk (GPU required)."""
    script = PIPELINE_DIR / "panderm" / "extract_features.py"
    if not script.exists():
        raise FileNotFoundError(f"Extract features script not found: {script}")
    cmd  = [sys.executable, str(script), "--mode", "frozen"]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def stage_evaluate(use_finetuned: bool = True):
    """Metrics + plots: frozen via logistic regression, finetune via the saved head predictions."""
    script = PIPELINE_DIR / "panderm" / "evaluate.py"
    mode   = "finetune" if use_finetuned else "frozen"
    cmd    = [sys.executable, str(script), "--mode", mode]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def stage_clinical_modality():
    """Run BioClinicalBERT to extract embeddings from pathology reports."""
    script = PIPELINE_DIR / "clinical" / "clinical_pipeline.py"
    if not script.exists():
        raise FileNotFoundError(f"Clinical pipeline script not found: {script}")

    cmd = [sys.executable, str(script)]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"\nEmbeddings saved to: {CLINICAL_DIR}")


def stage_fusion(use_finetuned: bool = True):
    """Late fusion: concatenate image + clinical text features → logistic regression."""
    script = PIPELINE_DIR / "fusion" / "late_fusion.py"
    if not script.exists():
        raise FileNotFoundError(f"Fusion script not found: {script}")

    mode = "finetune" if use_finetuned else "frozen"
    cmd  = [sys.executable, str(script), "--mode", mode]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ── Stage registry ────────────────────────────────────────────────────────

def _stage_map():
    """Build stage registry (evaluated at call time so cfg paths are resolved)."""
    return {
        "prepare": (
            stage_prepare,
            lambda: (OUTPUT_DIR / "dataset_manifest.csv").exists(),
        ),
        "segment": (
            stage_segment,
            lambda: SEGMENTED_DIR.exists() and any(SEGMENTED_DIR.iterdir()),
        ),
        "make_csv": (
            stage_make_csv,
            lambda: (CSV_DIR / SEGMENTATION_BACKEND / "panderm_finetuning_fold0.csv").exists(),
        ),
        "finetune": (
            stage_finetune,
            lambda: (FINETUNE_DIR / f"results_fold{N_FOLDS - 1}" /
                     f"test_predictions_fold{N_FOLDS - 1}.csv").exists(),
        ),
        "extract_features": (
            stage_extract_features,
            lambda: (OUTPUT_DIR / "features" / "frozen" / SEGMENTATION_BACKEND /
                     "patient_features_fold0.npy").exists(),
        ),
        "evaluate": (
            stage_evaluate,
            lambda: (OUTPUT_DIR / "kfold_results_finetuned.csv").exists(),
        ),
        "clinical_modality": (
            stage_clinical_modality,
            lambda: (CLINICAL_DIR / "clinical_embeddings.npy").exists(),
        ),
        "fusion": (
            stage_fusion,
            lambda: (OUTPUT_DIR / "fusion_results" / "fusion_kfold_results.csv").exists(),
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    stage_map = _stage_map()

    # Determine which stages to run
    if args.stage == "all":
        if args.no_finetune:
            to_run = ["prepare", "segment", "make_csv",
                      "extract_features", "evaluate",
                      "clinical_modality", "fusion"]
        else:
            # no extract_features: fine-tuned models are scored with their own head
            to_run = ["prepare", "segment", "make_csv",
                      "finetune", "evaluate",
                      "clinical_modality", "fusion"]
    else:
        to_run = [args.stage]

    print("=" * 60)
    print("MULTIMODAL AI FOR ONCOLOGY — PIPELINE")
    print("=" * 60)
    print(f"  Stages        : {', '.join(to_run)}")
    print(f"  Skip existing : {args.skip_existing}")
    print(f"  Data root     : {DATA_ROOT}")
    print(f"  Output dir    : {OUTPUT_DIR}")
    print(f"  Checkpoint    : {CHECKPOINT_LARGE}")

    total_start = time.time()

    use_finetuned = not args.no_finetune

    for stage_name in to_run:
        func, skip_check = stage_map[stage_name]

        if stage_name == "fusion":
            bound_func = lambda f=use_finetuned: stage_fusion(use_finetuned=f)
        elif stage_name == "evaluate":
            bound_func = lambda f=use_finetuned: stage_evaluate(use_finetuned=f)
        else:
            bound_func = func

        run_stage(
            name=stage_name,
            func=bound_func,
            skip_check=skip_check,
            skip_existing=args.skip_existing,
        )

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE  ({int(total // 60)}m {total % 60:.1f}s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
