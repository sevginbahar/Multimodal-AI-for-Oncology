"""
Restructure the ISIC subset into the folder layout run_pipeline.py expects
==========================================================================
`prepare_data.py`'s scanner needs DATA_ROOT/{class}/{patient_id}/*.jpg — one
subfolder per patient. ISIC downloads flat (isic/dn/ISIC_*.jpg, no per-patient
grouping), so this is a *restructure*, not a rename: it reads each patient_id
from metadata.csv and creates the missing subfolder level, hard-linking each
image into place (zero extra disk):

    <_PHD>/isic_pipeline/dermoscopy/{DN,MIA,Minsitu}/<group>/<isic_id>.jpg

<group> is the ISIC patient_id when present, else "solo_<isic_id>" (one image
per group) — this is the StratifiedGroupKFold grouping unit downstream.

Source layout expected (built by the `isic image download` commands, which
already filtered to image_type:dermoscopic AND
diagnosis_confirm_type:histopathology server-side — NOT re-checked here):
    <_PHD>/isic/dn/*.jpg   + metadata.csv
    <_PHD>/isic/mis/*.jpg  + metadata.csv
    <_PHD>/isic/mia/*.jpg  + metadata.csv

Class rename + the one filter not covered by the download query:
    dn  -> DN
    mis -> Minsitu
    mia -> MIA   (mel_thick_mm <= --mia-thick-max, default 1.0 = all T1;
                  use 0.8 for strict Stage IA)   AND mel_ulcer != True
                  (mel_ulcer was not part of the download search)

Usage:
    python isic_eval/build_isic_pipeline_layout.py                    # everything, unbalanced
    python isic_eval/build_isic_pipeline_layout.py --balance          # cap every class to the smallest class's group count
    python isic_eval/build_isic_pipeline_layout.py --balance-n 168    # cap every class to N groups
    python isic_eval/build_isic_pipeline_layout.py --mia-thick-max 0.8  # strict Stage IA instead of all-T1
    python isic_eval/build_isic_pipeline_layout.py --clean             # wipe & rebuild

Then:
    (edit config.py to the ISIC path block, or set $env:DATASET if using that variant)
    python run_pipeline.py --stage prepare
    python run_pipeline.py --stage segment              # or skip, per the segmentation ablation
    python run_pipeline.py --stage extract_features --no-finetune
    python run_pipeline.py --stage evaluate         --no-finetune
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RANDOM_SEED  # noqa: E402

_PHD = Path(r"C:\Users\bahar sevgin\Desktop\phd_data")
ISIC_ROOT = _PHD / "isic"
DEST_ROOT = _PHD / "isic_pipeline" / "dermoscopy"
FOLDER_TO_CLASS = {"dn": "DN", "mis": "Minsitu", "mia": "MIA"}


def select(folder: str, cls: str, mia_thick_max: float) -> pd.DataFrame:
    # image_type / diagnosis_confirm_type are NOT re-checked here — the `isic
    # image download --search` query already filtered to dermoscopic +
    # histopathology-confirmed server-side. Only apply filters that weren't
    # part of that query.
    m = pd.read_csv(ISIC_ROOT / folder / "metadata.csv")
    if cls == "MIA":
        if "mel_thick_mm" in m.columns:
            m = m[m["mel_thick_mm"].fillna(1e9) <= mia_thick_max]
        if "mel_ulcer" in m.columns:
            m = m[m["mel_ulcer"] != True]  # noqa: E712  (keep NaN/False)
    pid = m["patient_id"].astype("string").str.strip()
    m = m.assign(
        cls=cls,
        grp=pid.where(pid.notna() & (pid != ""), "solo_" + m["isic_id"].astype(str)),
        srcdir=str(ISIC_ROOT / folder),
    )
    return m[["isic_id", "cls", "grp", "srcdir"]]


def cap_class(df: pd.DataFrame, cls: str, n_groups: int) -> pd.DataFrame:
    """Keep at most n_groups whole groups of class `cls`; drop the rest."""
    sub = df[df["cls"] == cls]
    groups = sub["grp"].drop_duplicates()
    if len(groups) <= n_groups:
        return df
    keep = groups.sample(n=n_groups, random_state=RANDOM_SEED)
    return pd.concat([df[df["cls"] != cls], sub[sub["grp"].isin(keep)]], ignore_index=True)


def link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        return "exists"
    try:
        import os
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--balance", action="store_true",
                    help="cap every class to the smallest class's group count")
    ap.add_argument("--balance-n", type=int, default=None,
                    help="cap every class to N groups (implies --balance)")
    ap.add_argument("--mia-thick-max", type=float, default=1.0,
                    help="mel_thick_mm cutoff for MIA (1.0 = all T1, 0.8 = strict Stage IA)")
    ap.add_argument("--clean", action="store_true", help="wipe DEST_ROOT first")
    args = ap.parse_args()

    if args.clean and DEST_ROOT.exists():
        shutil.rmtree(DEST_ROOT)
        print(f"wiped {DEST_ROOT}")

    df = pd.concat(
        [select(f, c, args.mia_thick_max) for f, c in FOLDER_TO_CLASS.items()],
        ignore_index=True,
    )

    print("Before balancing:")
    print(df.groupby("cls").agg(images=("isic_id", "size"), groups=("grp", "nunique")))

    if args.balance or args.balance_n is not None:
        counts = df.groupby("cls")["grp"].nunique()
        target = args.balance_n if args.balance_n is not None else int(counts.min())
        print(f"\nBalancing every class to {target} groups "
              f"({'smallest class' if args.balance_n is None else 'explicit target'})")
        for cls in counts.index:
            df = cap_class(df, cls, target)

    stats = {"link": 0, "copy": 0, "exists": 0, "missing": 0}
    for r in df.itertuples(index=False):
        src = Path(r.srcdir) / f"{r.isic_id}.jpg"
        if not src.exists():
            stats["missing"] += 1
            continue
        dst_dir = DEST_ROOT / r.cls / r.grp
        dst_dir.mkdir(parents=True, exist_ok=True)
        stats[link_or_copy(src, dst_dir / f"{r.isic_id}.jpg")] += 1

    print(f"\nLayout: {DEST_ROOT}")
    print("After balancing:" if (args.balance or args.balance_n) else "Final:")
    print(df.groupby("cls").agg(images=("isic_id", "size"), groups=("grp", "nunique")))
    print(f"\n{stats}")
    print("\nNext: edit config.py to the ISIC path block, then")
    print("  python run_pipeline.py --stage prepare")
    print("  python run_pipeline.py --stage segment              # or skip, per the segmentation decision")
    print("  python run_pipeline.py --stage extract_features --no-finetune")
    print("  python run_pipeline.py --stage evaluate         --no-finetune")


if __name__ == "__main__":
    main()
