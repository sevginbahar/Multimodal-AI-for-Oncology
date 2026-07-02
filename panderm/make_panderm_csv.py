"""
Step 3: Convert dataset_manifest.csv → PanDerm fold CSVs
=========================================================
Generates one CSV per fold with columns: image, label, split
where split is train / val / test.

Fold-to-split mapping:
    test fold  N   → test
    val  fold  N+1 → val
    remaining  → train

Usage:
    python make_panderm_csv.py
    python make_panderm_csv.py --use-segmented   # use segmented image paths
"""
import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OUTPUT_DIR, SEGMENTED_DIR, DATA_ROOT,
    CSV_DIR, N_FOLDS,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-segmented", action="store_true",
                        help="Use segmented image paths instead of raw")
    args = parser.parse_args()

    manifest_path = OUTPUT_DIR / "dataset_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run prepare_data.py first."
        )

    df = pd.read_csv(manifest_path)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    for test_fold in range(N_FOLDS):
        val_fold = (test_fold + 1) % N_FOLDS

        def fold_to_split(fold, tf=test_fold, vf=val_fold):
            if fold == tf: return "test"
            if fold == vf: return "val"
            return "train"

        fold_df = df.copy()
        fold_df["split"] = fold_df["fold"].apply(fold_to_split)

        # Choose image column
        if args.use_segmented and "segmented_path" in fold_df.columns:
            fold_df["image"] = fold_df["segmented_path"]
        else:
            fold_df["image"] = fold_df["image_path"]

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
