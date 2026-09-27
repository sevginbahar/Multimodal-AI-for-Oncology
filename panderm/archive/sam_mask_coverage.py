"""
SAM mask coverage across the whole cohort — is SAM under-segmenting a
specific diagnosis class (e.g. DN's soft, ill-defined dysplastic borders)?

For every raw image in the manifest, runs the same centre-point-prompt
SAMSegmenter used by segment_lesions.py and records what fraction of the
RAW frame the mask covers (before any bounding-box crop/resize). Groups by
diagnosis and runs Kruskal-Wallis (omnibus) + pairwise Mann-Whitney
(Holm-corrected), matching this project's existing non-parametric
convention. Also saves a boxplot and a per-class near-median overlay
montage — this replaces the old one-image smoke_test_sam.py sanity check.

Run in sam2env (needs the real sam2 package):
    python panderm/sam_mask_coverage.py
    python panderm/sam_mask_coverage.py --n 100   # quick subsample first
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from segment_lesions import build_segmenter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import OUTPUT_DIR, RANDOM_SEED

RES_DIR = OUTPUT_DIR / "attention_comparison"


def save_boxplot(df: pd.DataFrame, classes: list, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = [df.loc[df["diagnosis"] == c, "mask_coverage"] for c in classes]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(data)
    ax.set_xticks(range(1, len(classes) + 1))
    ax.set_xticklabels(classes)
    ax.set_ylabel("mask coverage (fraction of raw frame)")
    ax.set_title("SAM mask coverage by diagnosis")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def save_montage(df: pd.DataFrame, classes: list, segmenter, out_path: Path) -> None:
    """One overlay per class, on the image closest to that class's OWN median
    coverage — not the true min/max. The true extremes (~0% / ~100%) turned
    out to be SAM's degenerate total-failure modes in EVERY class alike, which
    drowns out the actual per-class difference the stats found (that lives in
    the medians, e.g. DN's 13% vs Minsitu's 32%), so min/max isn't the useful
    comparison here."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 4))
    axes = np.atleast_1d(axes)

    for col_i, diagnosis in enumerate(classes):
        sub = df[df["diagnosis"] == diagnosis]
        med = sub["mask_coverage"].median()
        rec = sub.loc[(sub["mask_coverage"] - med).abs().idxmin()]
        img = np.array(Image.open(rec["image_path"]).convert("RGB"))
        mask = segmenter.segment(img)
        overlay = img.copy()
        overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        ax = axes[col_i]
        ax.imshow(overlay)
        ax.set_title(f"{diagnosis}  near-median  pid={rec['patient_id']}  "
                    f"cov={rec['mask_coverage']:.1%}", fontsize=9)
        ax.axis("off")
        print(f"  {diagnosis} near-median: {rec['image_path']}  coverage={rec['mask_coverage']:.1%}")

    plt.suptitle("SAM mask — representative (near-median) example per diagnosis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def save_worst_montage(df: pd.DataFrame, classes: list, segmenter, out_path: Path,
                       n_per_class: int = 1) -> None:
    """Per class, the N lowest- and N highest-coverage images — the actual
    worst failures (mask found ~nothing, or mask grabbed ~everything),
    as opposed to save_montage's near-median "typical case" picture."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(classes) * n_per_class, 2,
                             figsize=(8, 4 * len(classes) * n_per_class))
    axes = np.atleast_2d(axes)

    row_i = 0
    for diagnosis in classes:
        sub = df[df["diagnosis"] == diagnosis]
        lowest = sub.nsmallest(n_per_class, "mask_coverage")
        highest = sub.nlargest(n_per_class, "mask_coverage")
        for k in range(n_per_class):
            for col_i, (tag, rec) in enumerate([("lowest", lowest.iloc[k]), ("highest", highest.iloc[k])]):
                ax = axes[row_i, col_i]
                img = np.array(Image.open(rec["image_path"]).convert("RGB"))
                mask = segmenter.segment(img)
                overlay = img.copy()
                overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
                ax.imshow(overlay)
                ax.set_title(f"{diagnosis}  {tag}-coverage  pid={rec['patient_id']}  "
                            f"cov={rec['mask_coverage']:.1%}", fontsize=9)
                ax.axis("off")
                print(f"  {diagnosis} {tag}: {rec['image_path']}  coverage={rec['mask_coverage']:.1%}")
            row_i += 1

    plt.suptitle("SAM mask — worst cases per diagnosis (lowest vs. highest coverage)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def holm_correct(p_raw: "list[float]") -> "list[float]":
    """Holm-Bonferroni across a family of p-values, same method used in
    attention_comparison.py's friedman_and_posthoc."""
    order = np.argsort(p_raw)
    m = len(p_raw)
    holm = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p_raw[idx]
        running_max = max(running_max, adj)
        holm[idx] = min(running_max, 1.0)
    return holm.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="subsample size (default: whole manifest)")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--worst-n-per-class", type=int, default=1,
                    help="also save a montage of the N lowest- and N highest-coverage "
                         "images per class (the actual worst failures)")
    args = ap.parse_args()

    manifest = pd.read_csv(OUTPUT_DIR / "dataset_manifest.csv")
    if args.n:
        manifest = manifest.sample(n=min(args.n, len(manifest)), random_state=args.seed).reset_index(drop=True)
    print(f"Running SAM on {len(manifest)} images...")

    segmenter = build_segmenter("sam")
    rows = []
    for i, row in manifest.iterrows():
        img_path = Path(row["image_path"])
        if not img_path.exists():
            continue
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = segmenter.segment(img)
        rows.append({
            "image_path": row["image_path"],
            "patient_id": row.get("patient_id", row.get("folder_name", "?")),
            "diagnosis": row["diagnosis"],
            "mask_coverage": float(mask.mean()),
        })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(manifest)}")

    df = pd.DataFrame(rows)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RES_DIR / "sam_mask_coverage.csv", index=False)

    print("\nMask coverage (% of raw frame) by diagnosis:")
    summary = df.groupby("diagnosis")["mask_coverage"].agg(["mean", "median", "std", "count"])
    print(summary)
    summary.to_csv(RES_DIR / "sam_mask_coverage_summary.csv")

    classes = sorted(df["diagnosis"].unique())
    groups = [df.loc[df["diagnosis"] == c, "mask_coverage"] for c in classes]
    if len(classes) >= 3 and all(len(g) >= 2 for g in groups):
        h, p = stats.kruskal(*groups)
        print(f"\nKruskal-Wallis omnibus (does coverage differ by diagnosis at all?): "
             f"H={h:.3f}  p={p:.4f}")

        pairs, p_raw, deltas = [], [], []
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                a, b = classes[i], classes[j]
                u, wp = stats.mannwhitneyu(groups[i], groups[j], alternative="two-sided")
                pairs.append((a, b)); p_raw.append(wp)
                deltas.append(float(groups[i].mean() - groups[j].mean()))
        p_holm = holm_correct(p_raw)
        print("\nPairwise Mann-Whitney (Holm-corrected):")
        posthoc_rows = []
        for (a, b), wp, ph, d in zip(pairs, p_raw, p_holm, deltas):
            star = " *" if ph < 0.05 else ""
            print(f"  {a:<8} vs {b:<8}  mean_delta={d:+.4f}  p_raw={wp:.4f}  p_holm={ph:.4f}{star}")
            posthoc_rows.append({"a": a, "b": b, "mean_delta": d, "p_raw": wp, "p_holm": ph})
        pd.DataFrame(posthoc_rows).to_csv(RES_DIR / "sam_mask_coverage_pairwise.csv", index=False)

    save_boxplot(df, classes, RES_DIR / "sam_mask_coverage_boxplot.png")
    save_montage(df, classes, segmenter, RES_DIR / "sam_coverage_montage.png")
    save_worst_montage(df, classes, segmenter, RES_DIR / "sam_worst_coverage_montage.png",
                       n_per_class=args.worst_n_per_class)

    print(f"\nResults: {RES_DIR}")


if __name__ == "__main__":
    main()
