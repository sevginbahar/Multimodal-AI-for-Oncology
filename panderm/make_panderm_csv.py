"""
Step 3: Convert dataset_manifest.csv → PanDerm fold CSVs
=========================================================
Generates one CSV per fold with columns: image, label, split
where split is train / val / test.

Fold-to-split mapping:
    test fold  N   → test
    val  fold  N+1 → val
    remaining  → train

The image column follows config.SEGMENTATION_BACKEND: raw images for "none",
segmented_cache/<backend>/ crops otherwise (see config.input_image).

Usage:
    python make_panderm_csv.py
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OUTPUT_DIR, SEGMENTATION_BACKEND,
    CSV_DIR, N_FOLDS, input_image,
)

# Nested by backend — otherwise building the CSV for "color" silently
# overwrites "none"'s CSV at the same path, and a forgotten rebuild before
# switching backends would train on stale data without any error.
CSV_DIR = CSV_DIR / SEGMENTATION_BACKEND


def main():
    manifest_path = OUTPUT_DIR / "dataset_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run prepare_data.py first."
        )

    df = pd.read_csv(manifest_path)
    df["image"] = df["image_path"].map(input_image)
    missing = ~df["image"].map(lambda p: Path(p).exists())
    if missing.any():
        raise FileNotFoundError(
            f"{missing.sum()} of {len(df)} images for backend '{SEGMENTATION_BACKEND}' not found, "
            f"e.g. {df.loc[missing, 'image'].iloc[0]}\n"
            "Check DATA_ROOT in config.py, or run the segment stage for this backend.")
    print(f"Backend '{SEGMENTATION_BACKEND}': e.g. {df['image'].iloc[0]}")
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    for test_fold in range(N_FOLDS):
        val_fold = (test_fold + 1) % N_FOLDS

        def fold_to_split(fold, tf=test_fold, vf=val_fold):
            if fold == tf: return "test"
            if fold == vf: return "val"
            return "train"

        fold_df = df.copy()
        fold_df["split"] = fold_df["fold"].apply(fold_to_split)

        panderm_df = fold_df[["image", "label", "split"]]
        out_path   = CSV_DIR / f"panderm_finetuning_fold{test_fold}.csv"
        panderm_df.to_csv(out_path, index=False)

        counts = fold_df["split"].value_counts()
        print(f"Fold {test_fold}: train={counts.get('train', 0):4d}  "
              f"val={counts.get('val', 0):4d}  "
              f"test={counts.get('test', 0):4d}  → {out_path.name}")

    print(f"\nAll {N_FOLDS} CSVs saved to: {CSV_DIR}")


if __name__ == "__main__":
    main()
