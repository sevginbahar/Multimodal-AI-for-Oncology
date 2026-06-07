"""
Utility: Convert dataset_manifest.csv  →  panderm_finetuning.csv

The pipeline's manifest has:
    image_path, patient_id, folder_name, diagnosis, label, fold

PanDerm's run_class_finetuning.py expects:
    image, label, split        (where split is train / val / test)

Fold-to-split mapping (default):
    fold 0  →  test
    fold 1  →  val
    fold 2,3,4  →  train

You can override which fold is test/val with --test-fold / --val-fold.

Usage:
    python make_panderm_csv.py
    python make_panderm_csv.py --test-fold 0 --val-fold 1
    python make_panderm_csv.py --use-segmented   # use segmented image paths
"""
import argparse
import pandas as pd
from pathlib import Path
from config import OUTPUT_DIR, SEGMENTED_DIR, DATA_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-fold", type=int, default=0,
                        help="Fold number to use as test set (default: 0)")
    parser.add_argument("--val-fold", type=int, default=1,
                        help="Fold number to use as val set (default: 1)")
    parser.add_argument("--use-segmented", action="store_true",
                        help="Point image paths to segmented_cache/ instead of raw images")
    args = parser.parse_args()

    manifest = OUTPUT_DIR / "dataset_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest}\n"
            "Run prepare_data.py first."
        )

    df = pd.read_csv(manifest)

    # Assign split
    def fold_to_split(fold):
        if fold == args.test_fold:
            return "test"
        elif fold == args.val_fold:
            return "val"
        else:
            return "train"

    df["split"] = df["fold"].apply(fold_to_split)

    # Optionally remap paths to segmented cache
    if args.use_segmented:
        def remap_to_segmented(raw_path):
            p = Path(raw_path)
            # raw:       DATA_ROOT / class / patient_folder / image.jpg
            # segmented: SEGMENTED_DIR / class / patient_folder / image.jpg
            try:
                rel = p.relative_to(DATA_ROOT)
                return str(SEGMENTED_DIR / rel)
            except ValueError:
                return raw_path  # fallback: keep original

        df["image_path"] = df["image_path"].apply(remap_to_segmented)

    # Build PanDerm CSV
    panderm_df = df[["image_path", "label", "split"]].rename(
        columns={"image_path": "image"}
    )

    out_path = OUTPUT_DIR / "panderm_finetuning.csv"
    panderm_df.to_csv(out_path, index=False)

    # Summary
    counts = panderm_df["split"].value_counts()
    print(f"Saved: {out_path}")
    print(f"  train: {counts.get('train', 0)} images")
    print(f"  val:   {counts.get('val',   0)} images")
    print(f"  test:  {counts.get('test',  0)} images")
    print()
    print("Next step — run fine-tuning:")
    print(f"  cd F:\\multimodal_dermatology\\PanDerm\\classification")
    print(f"  python run_class_finetuning.py \\")
    print(f'      --csv_path "{out_path}" \\')
    print(f'      --root_path "" \\')
    print(f"      --nb_classes 3 \\")
    print(f"      ... (see STUDENT_GUIDE.md section 8.2 for full args)")


if __name__ == "__main__":
    main()
