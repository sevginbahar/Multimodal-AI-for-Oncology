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
    print(f"\n  ✓ Completed in {int(elapsed // 60)}m {elapsed % 60:.1f}s")


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
    """Segment lesions: LAB colour space + Otsu + morphological closing."""
    sys.path.insert(0, str(PIPELINE_DIR / "panderm"))
    from segment_lesions import ColorBasedSegmenter, process_dataset, generate_qc_montage

    manifest_csv = OUTPUT_DIR / "dataset_manifest.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Run 'prepare' first — not found: {manifest_csv}")

    segmenter = ColorBasedSegmenter(MORPH_KERNEL_SIZE, MIN_LESION_RATIO)
    df = process_dataset(
        manifest_csv, SEGMENTED_DIR, segmenter, IMAGE_SIZE, CROP_MARGIN
    )
    df.to_csv(manifest_csv, index=False)
    print(f"Manifest updated with segmented_path: {manifest_csv}")
    generate_qc_montage(df, segmenter, OUTPUT_DIR / "segmentation_qc_montage.png")


def stage_make_csv():
    """Convert manifest to one PanDerm-format CSV per fold."""
    import pandas as pd

    manifest_csv = OUTPUT_DIR / "dataset_manifest.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Run 'prepare' first — not found: {manifest_csv}")

    df = pd.read_csv(manifest_csv)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    for test_fold in range(N_FOLDS):
        val_fold = (test_fold + 1) % N_FOLDS

        def split_label(fold, tf=test_fold, vf=val_fold):
            if fold == tf: return "test"
            if fold == vf: return "val"
            return "train"

        fold_df = df.copy()
        fold_df["image"] = (
            fold_df["segmented_path"]
            if "segmented_path" in fold_df.columns
            else fold_df["image_path"]
        )
        fold_df["split"] = fold_df["fold"].apply(split_label)
        out = CSV_DIR / f"panderm_finetuning_fold{test_fold}.csv"
        fold_df[["image", "label", "split"]].to_csv(out, index=False)
        counts = fold_df["split"].value_counts()
        print(
            f"  Fold {test_fold}: train={counts.get('train', 0):4d}  "
            f"val={counts.get('val', 0):4d}  test={counts.get('test', 0):4d}  → {out.name}"
        )

    print(f"\nAll {N_FOLDS} CSVs saved to: {CSV_DIR}")


def stage_finetune():
    """Fine-tune PanDerm — one subprocess per fold."""
    script = PIPELINE_DIR / "panderm" / "panderm_finetuning.py"
    if not script.exists():
        raise FileNotFoundError(f"Fine-tuning script not found: {script}")

    for fold in range(N_FOLDS):
        fold_out = OUTPUT_DIR / f"results_fold{fold}"
        fold_out.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, str(script),
            "--fold",         str(fold),
            "--data_root",    str(DATA_ROOT),
            "--csv_dir",      str(CSV_DIR),
            "--output_dir",   str(fold_out),
            "--checkpoint",   str(CHECKPOINT_LARGE),
            "--panderm_repo", str(PANDERM_CLASS),
        ]
        print(f"\n  Fold {fold}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    print(f"\nAll folds done. Checkpoints: {OUTPUT_DIR}/results_fold{{0–{N_FOLDS-1}}}/checkpoint-best.pth")


def stage_extract_features(use_finetuned: bool = True):
    """Extract CLS features using PanDerm and save to disk (GPU required)."""
    sys.path.insert(0, str(PIPELINE_DIR / "panderm"))
    import extract_features as ef
    ef.FEATURES_DIR = FEATURES_DIR / ("finetuned" if use_finetuned else "pretrained")
    ef.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    ef.main()


def stage_evaluate():
    """Run logistic regression + generate all plots (CPU only)."""
    sys.path.insert(0, str(PIPELINE_DIR / "panderm"))
    import evaluate as ev
    ev.main()


def stage_clinical_modality():
    """Run BioClinicalBERT to extract embeddings from pathology reports."""
    script = PIPELINE_DIR / "clinical" / "clinical_pipeline.py"
    if not script.exists():
        raise FileNotFoundError(f"Clinical pipeline script not found: {script}")

    cmd = [
        sys.executable, str(script),
        "--output_dir", str(CLINICAL_DIR),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"\nEmbeddings saved to: {CLINICAL_DIR}")


def stage_fusion():
    """Late fusion: concatenate image + clinical text features → logistic regression."""
    script = PIPELINE_DIR / "fusion" / "late_fusion.py"
    if not script.exists():
        raise FileNotFoundError(f"Fusion script not found: {script}")

    # Use fine-tuned features if available, otherwise fall back to pretrained
    finetuned_dir  = OUTPUT_DIR / "features_finetuned"
    pretrained_dir = OUTPUT_DIR / "features_pretrained"
    features_dir   = finetuned_dir if finetuned_dir.exists() else pretrained_dir

    cmd = [
        sys.executable, str(script),
        "--image_features_dir",  str(features_dir),
        "--clinical_embeddings", str(CLINICAL_DIR / "clinical_embeddings.npy"),
        "--output_dir",          str(OUTPUT_DIR / "fusion_results"),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"\nFusion results saved to: {OUTPUT_DIR / 'fusion_results'}")


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
            lambda: (CSV_DIR / "panderm_finetuning_fold0.csv").exists(),
        ),
        "finetune": (
            stage_finetune,
            lambda: (OUTPUT_DIR / "results_fold0" / "checkpoint-best.pth").exists(),
        ),
        "extract_features": (
            stage_extract_features,
            lambda: (FEATURES_DIR / "patient_features_fold0.npy").exists(),
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
            to_run = ["prepare", "segment", "make_csv",
                      "finetune", "extract_features", "evaluate",
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

        # Pass use_finetuned flag into extract_features
        if stage_name == "extract_features":
            bound_func = lambda f=use_finetuned: stage_extract_features(use_finetuned=f)
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
