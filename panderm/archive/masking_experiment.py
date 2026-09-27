"""
Masking-robustness experiment — is attention more fragile on low-pigment
lesions than high-pigment ones (per Oscar's feedback)?

Masks mask_frac of the image in patch_size x patch_size patches (filled with
the image's own mean color, not black, to avoid the background-bias artifact
found in the segmentation crops), on a pigment-stratified sample of lesions,
then compares masked vs. unmasked attention within each group and the
*degradation* between groups.

Pigment score is always taken from the "color" crop's foreground saturation,
even when --mask-backend=none — scoring off the raw frame would pick images
by whole-image color (background skin, hair, ruler, vignette included), not
lesion pigment.

Reuses the frozen-encoder loader and attention-extraction code from
attention_comparison.py rather than duplicating it.

Usage:
    python panderm/masking_experiment.py
    python panderm/masking_experiment.py --mask-backend none
    python panderm/masking_experiment.py --mask-n-per-group 50 --mask-frac 0.3

CLI flags:
    --mask-backend      none/color/sam crop to test on         (default: color)
    --mask-n-per-group  images per pigment group (low + high)  (default: 30)
    --mask-frac         fraction of patches masked             (default: 0.2)
    --patch-size        mask patch size in pixels              (default: 16)
    --seed              RNG seed for which patches get masked  (default: 42)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from attention_comparison import (
    load_frozen_encoder, raw_attention_map, attention_metrics, cache_path, pigment_score, RES_DIR,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import OUTPUT_DIR, RANDOM_SEED


def build_pigment_groups(n_per_group: int, seed: int, score_backend: str = "color"
                         ) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Stratifies the full manifest by pigment_score and returns the
    n_per_group lowest- and highest-saturation images as (low, high).

    score_backend is always "color" (or another segmented backend), regardless
    of which backend the masking test itself runs on: pigment_score on
    backend="none" averages saturation over the WHOLE raw frame (background
    skin, hair, ruler, vignette included), which is a proxy for frame color,
    not lesion pigment. Scoring off the segmented crop's foreground keeps the
    low/high grouping about the lesion even when testing the "none" backend.
    """
    manifest = pd.read_csv(OUTPUT_DIR / "dataset_manifest.csv")
    manifest["pigment_score"] = manifest.apply(lambda r: pigment_score(r, score_backend), axis=1)
    manifest = manifest.dropna(subset=["pigment_score"])
    low = manifest.nsmallest(n_per_group, "pigment_score").reset_index(drop=True)
    high = manifest.nlargest(n_per_group, "pigment_score").reset_index(drop=True)
    return low, high


def run_masking_experiment(model, n_per_group: int, mask_frac: float,
                           patch_size: int, backend: str, seed: int) -> list:
    """Runs each sampled image both unmasked and masked; returns one record
    per (image, masked) pair, tagged with its pigment_group."""
    low, high = build_pigment_groups(n_per_group, seed)
    low = low.assign(pigment_group="low")
    high = high.assign(pigment_group="high")
    images = pd.concat([low, high], ignore_index=True)
    print(f"Pigment groups: low n={len(low)} (sat {low['pigment_score'].min():.1f}-{low['pigment_score'].max():.1f})  "
         f"high n={len(high)} (sat {high['pigment_score'].min():.1f}-{high['pigment_score'].max():.1f})")

    records = []
    for i, row in images.iterrows():
        p = cache_path(row, backend)
        if not p.exists():
            continue
        for masked in (False, True):
            native_grid, attn_map_224, resized, mask_grid = raw_attention_map(
                model, p, mask_frac=mask_frac if masked else 0.0,
                patch_size=patch_size, seed=seed + i)
            m = attention_metrics(native_grid, attn_map_224,
                                  resized if backend != "none" else None, mask_grid)
            m.update(image_path=row["image_path"], diagnosis=row["diagnosis"],
                     pigment_group=row["pigment_group"], pigment_score=row["pigment_score"],
                     masked=masked, patient_id=row.get("patient_id", row.get("folder_name", "?")))
            m["_attn"] = attn_map_224
            m["_resized"] = resized
            records.append(m)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(images)}")
    return records


def save_masking_overlay_grid(records: list, out_path: Path, n_per_group: int = 2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_image = {}
    for r in records:
        by_image.setdefault(r["image_path"], {"group": r["pigment_group"]})[r["masked"]] = r
    low_keys = [k for k, v in by_image.items() if v["group"] == "low"][:n_per_group]
    high_keys = [k for k, v in by_image.items() if v["group"] == "high"][:n_per_group]
    keys = low_keys + high_keys
    if not keys:
        return

    fig, axes = plt.subplots(len(keys), 2, figsize=(8, 4 * len(keys)))
    axes = np.atleast_2d(axes)
    for row_i, img_path in enumerate(keys):
        entry = by_image[img_path]
        for col_i, masked in enumerate([False, True]):
            ax = axes[row_i, col_i]
            rec = entry.get(masked)
            if rec is None:
                ax.axis("off"); continue
            resized = rec["_resized"]
            attn = rec["_attn"]
            heat = (attn - attn.min()) / (attn.max() - attn.min() + 1e-9)
            heat_color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(resized, 0.6, heat_color, 0.4, 0)
            bg = resized.max(axis=-1) <= 8
            overlay[bg] = 0
            ax.imshow(overlay)
            tag = "masked" if masked else "unmasked"
            ax.set_title(f"{entry['group']}-pigment  pid={rec.get('patient_id', '?')}  "
                        f"{tag}  ent={rec['entropy']:.2f}", fontsize=9)
            ax.axis("off")
    plt.suptitle("20%% patch masking — attention, low- vs high-pigment lesions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def run_masking_main(model, args) -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Masking experiment: {args.mask_frac:.0%} patches masked (patch_size={args.patch_size}), "
         f"backend='{args.mask_backend}', {args.mask_n_per_group} images/pigment-group")

    records = run_masking_experiment(model, args.mask_n_per_group, args.mask_frac,
                                     args.patch_size, args.mask_backend, args.seed)
    if not records:
        print("No records produced — nothing to compare."); return

    df = pd.DataFrame(records)
    plain_cols = [c for c in df.columns if not c.startswith("_")]
    df[plain_cols].to_csv(RES_DIR / "masking_metrics.csv", index=False)

    # fg_attention_fraction only exists for backends with a segmented crop
    # (backend="none" has no black background to define "foreground" against).
    metric_cols = [c for c in ["entropy", "top10_concentration", "fg_attention_fraction",
                               "attention_on_masked"] if c in df.columns]
    print("\nMasked vs. unmasked, per pigment group (mean):")
    print(df.groupby(["pigment_group", "masked"])[metric_cols].mean())

    print("\nWithin-group masked-vs-unmasked (paired Wilcoxon):")
    within_rows = []
    for group in ["low", "high"]:
        sub = df[df["pigment_group"] == group]
        for metric in metric_cols:
            wide = sub.pivot(index="image_path", columns="masked", values=metric).dropna()
            if len(wide) < 2 or True not in wide.columns or False not in wide.columns:
                continue
            w, p = stats.wilcoxon(wide[True], wide[False])
            delta = float((wide[True] - wide[False]).mean())
            within_rows.append({"pigment_group": group, "metric": metric, "n": len(wide),
                                "delta_masked_minus_unmasked": delta, "wilcoxon_p": p})
            print(f"  {group:<5} {metric:<22} delta={delta:+.4f}  p={p:.4f}")
    pd.DataFrame(within_rows).to_csv(RES_DIR / "masking_within_group_tests.csv", index=False)

    print("\nBetween-group: does masking degrade low-pigment lesions MORE than high-pigment ones?")
    between_rows = []
    for metric in metric_cols:
        wide = df.pivot_table(index=["image_path", "pigment_group"], columns="masked",
                              values=metric).dropna().reset_index()
        if True not in wide.columns or False not in wide.columns:
            continue
        wide["delta"] = wide[True] - wide[False]
        low_delta = wide.loc[wide["pigment_group"] == "low", "delta"]
        high_delta = wide.loc[wide["pigment_group"] == "high", "delta"]
        if len(low_delta) < 2 or len(high_delta) < 2:
            continue
        u, p = stats.mannwhitneyu(low_delta, high_delta, alternative="two-sided")
        print(f"  {metric:<22} low_delta_mean={low_delta.mean():+.4f}  "
             f"high_delta_mean={high_delta.mean():+.4f}  mannwhitney_p={p:.4f}")
        between_rows.append({"metric": metric, "low_delta_mean": float(low_delta.mean()),
                             "high_delta_mean": float(high_delta.mean()), "mannwhitney_p": p})
    pd.DataFrame(between_rows).to_csv(RES_DIR / "masking_between_group_tests.csv", index=False)

    save_masking_overlay_grid(records, RES_DIR / "masking_overlay_grid.png")
    print(f"\nResults: {RES_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-frac", type=float, default=0.2)
    ap.add_argument("--patch-size", type=int, default=16)
    ap.add_argument("--mask-n-per-group", type=int, default=30)
    ap.add_argument("--mask-backend", default="color", help="which cached crop to run the masking test on")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    model = load_frozen_encoder()
    run_masking_main(model, args)


if __name__ == "__main__":
    main()
