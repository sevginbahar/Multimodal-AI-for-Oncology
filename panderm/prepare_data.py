"""
Step 1: Scan the dermoscopy dataset and create a CSV manifest with
patient-level stratified k-fold splits.

Usage:
    python prepare_data.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from config import (
    DATA_ROOT, OUTPUT_DIR, CLASS_NAMES, CLASS_LABELS,
    IMAGE_EXTENSIONS, N_FOLDS, RANDOM_SEED,
)


def scan_dataset(data_root: Path) -> pd.DataFrame:
    """Walk dermoscopy/{class}/{patient_id}/*.JPG and build a manifest."""
    rows = []
    for class_name in CLASS_NAMES:
        label = CLASS_LABELS[class_name]
        class_dir = data_root / class_name
        if not class_dir.exists():
            print(f"  WARNING: class directory not found: {class_dir}")
            continue

        for patient_dir in sorted(class_dir.iterdir()):
            if not patient_dir.is_dir():
                continue

            # Extract numeric patient ID from folder name
            # (some folders have annotations, e.g. "13901116 3 de IA")
            folder_name = patient_dir.name
            patient_id = folder_name.split()[0]

            images = [
                f for f in patient_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            ]

            if not images:
                print(f"  Skipping empty folder: {class_name}/{folder_name}")
                continue

            for img_path in sorted(images):
                rows.append({
                    "image_path": str(img_path),
                    "patient_id": patient_id,
                    "folder_name": folder_name,
                    "diagnosis": class_name,
                    "label": label,
                })

    df = pd.DataFrame(rows)
    return df


def create_kfold_splits(df: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    """
    Assign each image to a fold using StratifiedGroupKFold.

    Groups = patient_id  (ensures no patient leaks across folds)
    Stratification = per-patient majority label

    For the 3 multi-class patients (same person, different diagnoses),
    all their images land in the same fold regardless of class.
    """
    # Build a per-patient label for stratification (majority vote)
    patient_label = (
        df.groupby("patient_id")["label"]
        .agg(lambda x: x.value_counts().index[0])
    )

    # Map back to image rows
    groups = df["patient_id"].values
    strat_labels = df["patient_id"].map(patient_label).values

    # Create fold assignments
    df["fold"] = -1 # creating the fold column
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed) #creating the splitter

    for fold_idx, (_, test_idx) in enumerate(sgkf.split(df, strat_labels, groups)):
        df.loc[df.index[test_idx], "fold"] = fold_idx

    assert (df["fold"] == -1).sum() == 0, "Some images were not assigned a fold!"
    return df


def validate_no_leakage(df: pd.DataFrame, n_folds: int) -> bool:
    """Verify that no patient appears in both train and test for any fold."""
    ok = True
    for fold_idx in range(n_folds):
        test_patients = set(df.loc[df["fold"] == fold_idx, "patient_id"])
        train_patients = set(df.loc[df["fold"] != fold_idx, "patient_id"])
        overlap = test_patients & train_patients
        if overlap:
            print(f"  LEAK in fold {fold_idx}: patients in both sets: {overlap}")
            ok = False
    return ok


def print_statistics(df: pd.DataFrame, n_folds: int):
    """Print comprehensive dataset and split statistics."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    n_images = len(df)
    n_patients = df["patient_id"].nunique()

    print(f"\nTotal images:   {n_images}")
    print(f"Total patients: {n_patients}")

    # Per-class stats
    print(f"\n{'Class':<12} {'Patients':>10} {'Images':>10} {'Imgs/pat (median)':>20}")
    print("-" * 55)
    for cls in CLASS_NAMES:
        sub = df[df["diagnosis"] == cls]
        n_pat = sub["patient_id"].nunique()
        n_img = len(sub)
        median_per_pat = sub.groupby("patient_id").size().median()
        print(f"{cls:<12} {n_pat:>10} {n_img:>10} {median_per_pat:>20.0f}")

    # Multi-class patients
    pat_classes = df.groupby("patient_id")["diagnosis"].nunique()
    multi = pat_classes[pat_classes > 1]
    if len(multi) > 0:
        print(f"\nMulti-class patients ({len(multi)}):")
        for pid in multi.index:
            classes = df.loc[df["patient_id"] == pid, "diagnosis"].unique()
            print(f"  {pid}: {', '.join(classes)}")

    # Per-fold stats
    print(f"\n{'Fold':<6}", end="")
    for cls in CLASS_NAMES:
        print(f" {cls + '_pat':>10} {cls + '_img':>10}", end="")
    print(f" {'Total_pat':>10} {'Total_img':>10}")
    print("-" * 80)
    for fold_idx in range(n_folds):
        test = df[df["fold"] == fold_idx]
        print(f"{fold_idx:<6}", end="")
        for cls in CLASS_NAMES:
            sub = test[test["diagnosis"] == cls]
            print(f" {sub['patient_id'].nunique():>10} {len(sub):>10}", end="")
        print(f" {test['patient_id'].nunique():>10} {len(test):>10}")

    print("=" * 60)


def main():
    print("Step 1: Preparing dataset manifest with k-fold splits")
    print(f"  Data root: {DATA_ROOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Scan dataset
    print("\nScanning dataset...")
    df = scan_dataset(DATA_ROOT)
    print(f"  Found {len(df)} images from {df['patient_id'].nunique()} patients")

    # 2. Create k-fold splits
    print(f"\nCreating {N_FOLDS}-fold patient-level splits...")
    df = create_kfold_splits(df, N_FOLDS, RANDOM_SEED)

    # 3. Validate no leakage
    print("\nValidating no data leakage...")
    if validate_no_leakage(df, N_FOLDS):
        print("  OK: No patient leakage detected across folds.")
    else:
        print("  ERROR: Data leakage detected! Check the output above.")

    # 4. Save manifest
    csv_path = OUTPUT_DIR / "dataset_manifest.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nManifest saved to: {csv_path}")

    # 5. Print statistics
    print_statistics(df, N_FOLDS)


if __name__ == "__main__":
    main()
