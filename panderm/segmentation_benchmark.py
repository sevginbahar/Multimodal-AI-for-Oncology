"""
Segmentation benchmark against expert masks — ISIC 2018 Task 1
================================================================
Runs every segmenter (Otsu v1/v2, SAM2 v1, SAM2 v1 + preprocessing, SAM2 v2)
on the ISIC 2018 Task 1 images and compares each mask with the expert
ground truth. None of these segmenters is trained on ISIC, so the training
split is a fair test set for them.

Questions answered (one function each):
    B1  accuracy_summary()       How accurate is each method? (IoU, Dice, ISIC thresholded Jaccard,
                                 sensitivity, specificity, area ratio, failure = IoU < 0.65)
    B2  pairwise_accuracy()      Is method A more accurate than B on the same images?
                                 (Wilcoxon on IoU, bootstrap CI of the mean difference, McNemar on failure)
    B3  artifact_subgroups()     Do hair removal and field-of-view detection help where they should?
                                 (IoU / failure in heavy-hair vs other and vignette vs no-vignette images;
                                 paired v1 -> v2 change within each subgroup)
    B4  mask_agreement()         How similar are two methods' masks? (IoU between predicted masks)
    B5  disagreement_montages()  Where do two methods disagree? (lowest-agreement images,
                                 expert outline white, method A green, method B magenta)
    B6  failure_montages()       What do each method's worst failures look like?
    B7  qc_montage()             What does typical behaviour look like? (random images, all methods side by side)
    B8  iou_distribution()       How are images spread over IoU levels? (count / % per IoU bin and below 0.5 / 0.6 / 0.65 / 0.7)
    B9  failure_overlap()        Do methods fail on the same images? (both / only A / only B / neither)
    B10 contrast_heterogeneity() Does Otsu win on dark (high-contrast) lesions and SAM lose on heterogeneous
                                 ones? (3 x 3 grid of contrast x heterogeneity thirds, paired Otsu v2 - SAM v2
                                 difference, Spearman correlations; lesion measures from the expert mask)
    All tables are also collected in benchmark_summary.xlsx (rebuild alone with --summary-only).

Predicted masks are cached as PNGs (one per image and method), so an
interrupted run resumes and added methods never re-segment the others.

Inputs (defaults under PhD_Data/Data, override with --images / --gt):
    ISIC2018_Task1-2_Training_Input/ISIC2018_Task1-2_Training_Input/ISIC_*.jpg
    ISIC2018_Task1_Training_GroundTruth/**/ISIC_*_segmentation.png   (unzip the GroundTruth zip first)

Outputs (PhD_Data/isic2018_benchmark/<split>/, split = train or test):
    masks/<method>/ISIC_*.png        cached predicted masks
    preprocessing_stats.csv          hair fraction + field-of-view fraction per image
    per_image_masks.csv              one row per image x method: all metrics + segmenter QC
    b1_accuracy_summary.csv, b2_pairwise_accuracy.csv, b3_artifact_subgroups.csv,
    b4_mask_agreement_summary.csv, b4_mask_agreement_images.csv
    q6_disagreement_<a>_vs_<b>.png + .csv, b6_failures_<method>.png, b7_qc_montage.png,
    b1_iou_distribution.png

Official test split (--split test): the 1,000 ISIC 2018 Task 1 test images, outputs in
isic2018_benchmark/test/. Needed for any method trained on the training split (PanDerm-seg),
so trained and untrained methods are scored on the same unseen images at full resolution.

Usage:
    python panderm/segmentation_benchmark.py --n 500          # quick first run on a random subset
    python panderm/segmentation_benchmark.py                  # all 2,594 images (resumes from the cache)
    python panderm/segmentation_benchmark.py --methods color_v2 sam_v2
    # PanDerm-seg vs Otsu v2 vs SAM v2 on the test split:
    python panderm/segment_panderm_seg.py --isic-test --ckpt <ckpt>     # dermseg env: makes the PanDerm-seg masks
    python panderm/segmentation_benchmark.py --split test --methods color_v2 sam_v2 panderm_seg
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from segment_lesions import build_segmenter, to_work_res, dullrazor, field_of_view_mask
from segmentation_stats import wilson_ci, holm, iou_dice

PHD = Path(r"C:\Users\qp263329\OneDrive - Queen Mary, University of London\Desktop\PhD_Data")
# Official ISIC 2018 Task 1 splits: (images, ground truth, output folder).
# Outputs: isic2018_benchmark/train/ and isic2018_benchmark/test/.
SPLITS = {
    "train": (PHD / "Data/ISIC2018_Task1-2_Training_Input/ISIC2018_Task1-2_Training_Input",
              PHD / "Data/ISIC2018_Task1_Training_GroundTruth", PHD / "isic2018_benchmark" / "train"),
    "test": (PHD / "Data/ISIC2018_Task1-2_Test_Input/ISIC2018_Task1-2_Test_Input",
             PHD / "Data/ISIC2018_Task1_Test_GroundTruth", PHD / "isic2018_benchmark" / "test"),
}
OUT = SPLITS["train"][2]   # set from --split in main()

METHODS = {"color": "Otsu v1", "color_v2": "Otsu v2", "sam": "SAM v1",
           "sam_v1_pre": "SAM v1 + preproc", "sam_v2": "SAM v2", "panderm_seg": "PanDerm-seg"}
# Trained on the ISIC 2018 TRAINING split, so only valid on --split test. Masks are made in the
# dermseg env by segment_panderm_seg.py --isic-test (this script only loads them from the cache).
TRAINED_ON_ISIC_TRAIN = {"panderm_seg"}
V1_V2 = [("color", "color_v2"), ("sam", "sam_v2"), ("sam_v1_pre", "sam_v2")]
FAIL_IOU = 0.65          # ISIC 2018 thresholded-Jaccard cut-off
SMALL = 256              # long side for method-vs-method agreement
MONTAGE = 512            # long side of montage tiles


# ── Data and masks ───────────────────────────────────────────────────────

def find_pairs(images: Path, gt: Path) -> pd.DataFrame:
    """Image / ground-truth pairs matched by ISIC id."""
    masks = {p.name.replace("_segmentation.png", ""): p for p in gt.rglob("*_segmentation.png")}
    if not masks:
        raise SystemExit(f"No *_segmentation.png under {gt} — unzip ISIC2018_Task1_Training_GroundTruth.zip there.")
    rows = [{"isic_id": p.stem, "image_path": str(p), "gt_path": str(masks[p.stem])}
            for p in sorted(images.glob("ISIC_*.jpg")) if p.stem in masks]
    print(f"{len(rows)} image / ground-truth pairs ({len(masks)} masks found)")
    return pd.DataFrame(rows)


def predicted_mask(method: str, row, image: np.ndarray, seg_cache: dict) -> "tuple[np.ndarray, dict]":
    """Cached mask if present, else segment once and cache. Returns (mask, segmenter QC info)."""
    p = OUT / "masks" / method / f"{row.isic_id}.png"
    if p.exists():
        return np.array(Image.open(p)) > 0, {}
    if method in TRAINED_ON_ISIC_TRAIN:
        raise SystemExit(f"No cached '{method}' mask for {row.isic_id} in {p.parent}. Make them first "
                         f"(dermseg env): python panderm/segment_panderm_seg.py --isic-test --ckpt <checkpoint>")
    if method not in seg_cache:
        print(f"  loading segmenter '{method}'")
        seg_cache[method] = build_segmenter(method)
    seg = seg_cache[method]
    m = seg.segment(image) > 0
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(m.astype(np.uint8) * 255).save(p)
    return m, dict(getattr(seg, "last_info", {}) or {})


def preprocessing_stats(pairs: pd.DataFrame) -> pd.DataFrame:
    """Hair fraction and field-of-view fraction per image (the v2 preprocessing, method-independent).
    Cached in preprocessing_stats.csv."""
    path = OUT / "preprocessing_stats.csv"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["isic_id", "hair_frac", "fov_frac"])
    todo = pairs[~pairs.isic_id.isin(done.isic_id)]
    rows = []
    for i, r in enumerate(todo.itertuples(), 1):
        work = to_work_res(np.array(Image.open(r.image_path).convert("RGB")))
        clean, hair = dullrazor(work)
        rows.append({"isic_id": r.isic_id, "hair_frac": float(hair.mean()),
                     "fov_frac": float(field_of_view_mask(clean).mean())})
        if i % 200 == 0:
            print(f"    preprocessing stats {i}/{len(todo)}")
    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(path, index=False)
    return out


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    tp, fp = (pred & gt).sum(), (pred & ~gt).sum()
    fn, tn = (~pred & gt).sum(), (~pred & ~gt).sum()
    iou, dice = iou_dice(pred, gt)
    return {"iou": iou, "dice": dice, "thresholded_jaccard": iou if iou >= FAIL_IOU else 0.0,
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "area_ratio": pred.sum() / gt.sum() if gt.sum() else np.nan,
            "mask_frac_of_frame": float(pred.mean()), "failure": iou < FAIL_IOU}


def small(m: np.ndarray, side: int) -> np.ndarray:
    h, w = m.shape
    s = side / max(h, w)
    return cv2.resize(m.astype(np.uint8), (round(w * s), round(h * s)), interpolation=cv2.INTER_NEAREST) > 0


# ── B1 ───────────────────────────────────────────────────────────────────

def accuracy_summary(df: pd.DataFrame) -> pd.DataFrame:
    """B1 — How accurate is each method against the expert masks?
    Mean / median / IQR of IoU and Dice, ISIC thresholded-Jaccard mean (official 2018 score),
    failure rate (IoU < 0.65) with 95% Wilson CI, mean sensitivity and specificity,
    median area ratio (> 1 = over-segmentation)."""
    rows = []
    for m, g in df.groupby("method", sort=False):
        k, n = int(g.failure.sum()), len(g)
        lo, hi = wilson_ci(k, n)
        rows.append({"method": METHODS[m], "n_images": n,
                     "iou_mean": g.iou.mean(), "iou_median": g.iou.median(),
                     "iou_q25": g.iou.quantile(.25), "iou_q75": g.iou.quantile(.75),
                     "dice_mean": g.dice.mean(), "dice_median": g.dice.median(),
                     "thresholded_jaccard_mean": g.thresholded_jaccard.mean(),
                     "failure_rate": k / n, "failure_ci95_low": lo, "failure_ci95_high": hi,
                     "sensitivity_mean": g.sensitivity.mean(), "specificity_mean": g.specificity.mean(),
                     "area_ratio_median": g.area_ratio.median()})
    return pd.DataFrame(rows)


def iou_plot(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = [m for m in METHODS if m in df.method.unique()]
    fig, ax = plt.subplots(figsize=(1.8 * len(order) + 2, 4.5))
    ax.boxplot([df.loc[df.method == m, "iou"] for m in order], showfliers=False)
    for i, m in enumerate(order, 1):
        y = df.loc[df.method == m, "iou"].to_numpy()
        ax.scatter(np.random.default_rng(0).normal(i, 0.06, len(y)), y, s=3, alpha=0.25, color="0.3")
    ax.axhline(FAIL_IOU, color="tab:red", ls="--", lw=1)
    ax.text(len(order) + 0.45, FAIL_IOU, "failure < 0.65", color="tab:red", fontsize=8, va="center")
    ax.set_xticks(range(1, len(order) + 1), [METHODS[m] for m in order])
    ax.set_ylabel("IoU with expert mask")
    ax.set_title("ISIC 2018 Task 1 — IoU per image")
    fig.tight_layout()
    fig.savefig(OUT / "b1_iou_distribution.png", dpi=130)
    plt.close(fig)


# ── B2 ───────────────────────────────────────────────────────────────────

def pairwise_accuracy(df: pd.DataFrame, n_boot: int, rng) -> pd.DataFrame:
    """B2 — Is method A more accurate than B on the same images?
    Paired per image: Wilcoxon signed-rank on IoU, bootstrap 95% CI of the mean IoU
    difference (images resampled), and exact McNemar on failure (IoU < 0.65,
    discordant pairs only). Holm-corrected within each test family."""
    wide = df.pivot(index="isic_id", columns="method", values="iou")
    fail = df.pivot(index="isic_id", columns="method", values="failure").astype(bool)
    rows = []
    for a, b in combinations([m for m in METHODS if m in wide.columns], 2):
        d = (wide[a] - wide[b]).dropna().to_numpy()
        boot = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]
        w = stats.wilcoxon(wide[a], wide[b]) if np.any(d != 0) else None
        only_a, only_b = int((fail[a] & ~fail[b]).sum()), int((~fail[a] & fail[b]).sum())
        mc = stats.binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else np.nan
        rows.append({"method_a": METHODS[a], "method_b": METHODS[b], "n_images": len(d),
                     "mean_iou_a": wide[a].mean(), "mean_iou_b": wide[b].mean(),
                     "mean_iou_difference_a_minus_b": d.mean(),
                     "diff_ci95_low": np.percentile(boot, 2.5), "diff_ci95_high": np.percentile(boot, 97.5),
                     "share_a_better": (d > 0).mean(), "p_wilcoxon": w.pvalue if w else np.nan,
                     "fails_only_a": only_a, "fails_only_b": only_b, "p_mcnemar": mc})
    out = pd.DataFrame(rows)
    out["p_wilcoxon_holm"] = holm(out.p_wilcoxon.tolist())
    out["p_mcnemar_holm"] = holm(out.p_mcnemar.tolist())
    return out


# ── B3 ───────────────────────────────────────────────────────────────────

def artifact_subgroups(df: pd.DataFrame, pre: pd.DataFrame) -> pd.DataFrame:
    """B3 — Do hair removal and field-of-view detection help where they should?
    Subgroups (image level, from the v2 preprocessing):
      heavy hair  hair fraction in the top quartile of images
      vignette    field of view < 99% of the frame (dark scope rim / corners detected)
    Part 1: per method, median IoU and failure rate inside vs outside each subgroup,
            Mann-Whitney U on IoU (different images, so unpaired).
    Part 2: per v1 -> v2 pair, the paired IoU change (v2 - v1) inside each subgroup,
            Wilcoxon signed-rank. If the preprocessing works, the gain should be
            larger inside 'heavy hair' / 'vignette' than outside. Holm over all p-values."""
    pre = pre.assign(heavy_hair=pre.hair_frac >= pre.hair_frac.quantile(0.75), vignette=pre.fov_frac < 0.99)
    d = df.merge(pre[["isic_id", "heavy_hair", "vignette"]], on="isic_id")
    rows = []
    for flag in ("heavy_hair", "vignette"):
        for m, g in d.groupby("method", sort=False):
            yes, no = g[g[flag]], g[~g[flag]]
            p = stats.mannwhitneyu(yes.iou, no.iou).pvalue if len(yes) and len(no) else np.nan
            rows.append({"part": "1 method in vs out", "subgroup": flag, "method": METHODS[m],
                         "n_in": len(yes), "n_out": len(no),
                         "median_iou_in": yes.iou.median(), "median_iou_out": no.iou.median(),
                         "failure_rate_in": yes.failure.mean(), "failure_rate_out": no.failure.mean(),
                         "p_value": p})
        wide = d.pivot(index="isic_id", columns="method", values="iou").join(pre.set_index("isic_id")[[flag]])
        for v1, v2 in V1_V2:
            if v1 not in wide or v2 not in wide:
                continue
            for inside in (True, False):
                s = wide[wide[flag] == inside]
                delta = (s[v2] - s[v1]).dropna()
                p = stats.wilcoxon(s[v2], s[v1]).pvalue if (delta != 0).any() else np.nan
                rows.append({"part": "2 paired v1->v2 change", "subgroup": f"{flag}={'yes' if inside else 'no'}",
                             "method": f"{METHODS[v1]} -> {METHODS[v2]}", "n_in": len(delta),
                             "median_iou_change": delta.median(), "mean_iou_change": delta.mean(),
                             "share_improved": (delta > 0).mean(), "p_value": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p_value.tolist())
    return out


# ── B4 / B5 / B6 ─────────────────────────────────────────────────────────

def mask_agreement(pairs: pd.DataFrame, smalls: dict) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """B4 — How similar are two methods' masks on the same image? IoU and Dice between
    the predicted masks (not against the expert); median / IQR per pair."""
    rows = []
    for a, b in combinations([m for m in METHODS if m in smalls], 2):
        for iid in pairs.isic_id:
            iou, dice = iou_dice(smalls[a][iid], smalls[b][iid])
            rows.append({"isic_id": iid, "method_a": METHODS[a], "method_b": METHODS[b], "iou": iou, "dice": dice})
    img = pd.DataFrame(rows)
    summ = img.groupby(["method_a", "method_b"], sort=False).agg(
        n_images=("iou", "size"), iou_median=("iou", "median"), iou_q25=("iou", lambda s: s.quantile(.25)),
        iou_q75=("iou", lambda s: s.quantile(.75)), dice_median=("dice", "median"),
        share_iou_below_0_5=("iou", lambda s: (s < 0.5).mean())).reset_index()
    return img, summ


def _tile(ax, image_path: str, outlines: list, title: str):
    """Image with outlines [(mask, rgb), ...] drawn at montage resolution."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    s = MONTAGE / max(w, h)
    img = np.array(img.resize((round(w * s), round(h * s))))
    for m, colour in outlines:
        m = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cnt, -1, colour, 2)
    ax.imshow(img)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _load_full(method: str, iid: str) -> np.ndarray:
    return np.array(Image.open(OUT / "masks" / method / f"{iid}.png")) > 0


def disagreement_montages(pairs: pd.DataFrame, agree: pd.DataFrame, per_image: pd.DataFrame, n: int) -> None:
    """B5 — Where do two methods disagree? For every pair of methods: the n images with the
    lowest IoU between their masks. Expert outline white, method A green, method B magenta;
    each tile states both methods' IoU with the expert, so it shows which one was right."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = pairs.set_index("isic_id")
    vs_gt = per_image.pivot(index="isic_id", columns="method", values="iou")
    key = {v: k for k, v in METHODS.items()}
    for (la, lb), g in agree.groupby(["method_a", "method_b"], sort=False):
        a, b = key[la], key[lb]
        worst = g.nsmallest(n, "iou").assign(iou_a_vs_expert=lambda x: x.isic_id.map(vs_gt[a]),
                                              iou_b_vs_expert=lambda x: x.isic_id.map(vs_gt[b]))
        worst.to_csv(OUT / f"q6_disagreement_{a}_vs_{b}.csv", index=False)
        cols = 4
        fig, axes = plt.subplots(int(np.ceil(len(worst) / cols)), cols, figsize=(4.2 * cols, 3.6 * np.ceil(len(worst) / cols)))
        axes = np.atleast_1d(axes).ravel()
        for ax, r in zip(axes, worst.itertuples()):
            gt = np.array(Image.open(paths.at[r.isic_id, "gt_path"])) > 127
            _tile(ax, paths.at[r.isic_id, "image_path"],
                  [(gt, (255, 255, 255)), (_load_full(a, r.isic_id), (0, 220, 0)), (_load_full(b, r.isic_id), (255, 0, 255))],
                  f"{r.isic_id}  A vs B IoU={r.iou:.2f}\nvs expert: {la} {r.iou_a_vs_expert:.2f} | {lb} {r.iou_b_vs_expert:.2f}")
        for ax in axes[len(worst):]:
            ax.axis("off")
        fig.suptitle(f"Lowest agreement: {la} (green) vs {lb} (magenta); expert = white", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / f"q6_disagreement_{a}_vs_{b}.png", dpi=110)
        plt.close(fig)


def failure_montages(pairs: pd.DataFrame, per_image: pd.DataFrame, n: int) -> None:
    """B6 — What do each method's worst failures look like? The n lowest-IoU images per
    method: expert outline white, prediction green."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = pairs.set_index("isic_id")
    for m, g in per_image.groupby("method", sort=False):
        worst = g.nsmallest(n, "iou")
        cols = 4
        fig, axes = plt.subplots(int(np.ceil(len(worst) / cols)), cols, figsize=(4.2 * cols, 3.6 * np.ceil(len(worst) / cols)))
        axes = np.atleast_1d(axes).ravel()
        for ax, r in zip(axes, worst.itertuples()):
            gt = np.array(Image.open(paths.at[r.isic_id, "gt_path"])) > 127
            _tile(ax, paths.at[r.isic_id, "image_path"], [(gt, (255, 255, 255)), (_load_full(m, r.isic_id), (0, 220, 0))],
                  f"{r.isic_id}  IoU={r.iou:.2f}  area ratio={r.area_ratio:.2f}")
        for ax in axes[len(worst):]:
            ax.axis("off")
        fig.suptitle(f"{METHODS[m]} — worst {len(worst)} images (expert white, prediction green)", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / f"b6_failures_{m}.png", dpi=110)
        plt.close(fig)


def qc_montage(pairs: pd.DataFrame, per_image: pd.DataFrame, methods: list, n: int, seed: int) -> None:
    """B7 — What does typical behaviour look like? A RANDOM sample of n images (not the
    worst ones): one row per image, first column the raw image with the expert outline,
    then one column per method (expert white, prediction green, IoU in the title)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sample = pairs.sample(n=min(n, len(pairs)), random_state=seed).reset_index(drop=True)
    iou = per_image.pivot(index="isic_id", columns="method", values="iou")
    cols = 1 + len(methods)
    fig, axes = plt.subplots(len(sample), cols, figsize=(3.2 * cols, 2.8 * len(sample)))
    axes = np.atleast_2d(axes)
    for i, r in enumerate(sample.itertuples()):
        gt = np.array(Image.open(r.gt_path)) > 127
        _tile(axes[i, 0], r.image_path, [(gt, (255, 255, 255))], f"{r.isic_id}\nexpert (white)")
        for j, m in enumerate(methods, 1):
            _tile(axes[i, j], r.image_path, [(gt, (255, 255, 255)), (_load_full(m, r.isic_id), (0, 220, 0))],
                  f"{METHODS[m]}  IoU={iou.at[r.isic_id, m]:.2f}")
    fig.suptitle("Segmentation QC — random images (expert white, prediction green)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "b7_qc_montage.png", dpi=110)
    plt.close(fig)


IOU_BINS = [0, 0.1, 0.3, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0 + 1e-9]
IOU_LABELS = ["0-0.1", "0.1-0.3", "0.3-0.5", "0.5-0.6", "0.6-0.65", "0.65-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
IOU_CUTOFFS = [("below", 0.5), ("below", 0.6),
               ("below", 0.65), ("at or above", 0.65),     # ISIC 2018 acceptability threshold
               ("below", 0.7), ("at or above", 0.7)]       # good segmentations


def iou_distribution(per_image: pd.DataFrame) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """B8 — How are images spread over IoU levels?
    (a) count and % of images per IoU bin, per method (bins are [low, high));
    (b) count and % of images below each cut-off (0.5, 0.6, 0.65 = ISIC failure, 0.7)
        and at or above 0.7 (good segmentations)."""
    bins, below = [], []
    for m, g in per_image.groupby("method", sort=False):
        c = pd.cut(g.iou, IOU_BINS, labels=IOU_LABELS, right=False).value_counts().reindex(IOU_LABELS, fill_value=0)
        for lab, k in c.items():
            bins.append({"method": METHODS[m], "iou_range": lab, "images": int(k), "percent": round(100 * k / len(g), 1)})
        for direction, t in IOU_CUTOFFS:
            k = int((g.iou < t).sum() if direction == "below" else (g.iou >= t).sum())
            lo, hi = wilson_ci(k, len(g))
            below.append({"method": METHODS[m], "direction": direction, "iou": t, "images": k, "of_total": len(g),
                          "percent": round(100 * k / len(g), 1),
                          "ci95_low_%": round(100 * lo, 1), "ci95_high_%": round(100 * hi, 1)})
    return pd.DataFrame(bins), pd.DataFrame(below)


def failure_overlap(per_image: pd.DataFrame) -> pd.DataFrame:
    """B9 — For each pair of methods: images where both / only A / only B / neither fail (IoU < 0.65)."""
    fail = per_image.pivot(index="isic_id", columns="method", values="iou") < FAIL_IOU
    rows = []
    for a, b in combinations([m for m in METHODS if m in fail.columns], 2):
        n = len(fail)
        for lab, mask in (("both fail", fail[a] & fail[b]), (f"only {METHODS[a]} fails", fail[a] & ~fail[b]),
                          (f"only {METHODS[b]} fails", ~fail[a] & fail[b]), ("both succeed", ~fail[a] & ~fail[b])):
            rows.append({"method_a": METHODS[a], "method_b": METHODS[b], "outcome": lab,
                         "images": int(mask.sum()), "percent": round(100 * mask.sum() / n, 1)})
    return pd.DataFrame(rows)


TERTILES = ["low", "medium", "high"]


def lesion_characteristics(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per image, from the EXPERT mask (cached in lesion_characteristics.csv):
      contrast        median lightness (LAB L, 0-255) of a ring of skin around the lesion
                      minus the lesion's median lightness; higher = lesion darker than its skin
      heterogeneity   standard deviation of lightness inside the lesion
      colour_heterogeneity  mean LAB distance of lesion pixels from the lesion's mean colour
    Measured at the 1024 px working scale on the original image (no hair removal)."""
    path = OUT / "lesion_characteristics.csv"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["isic_id"])
    todo = pairs[~pairs.isic_id.isin(done.isic_id)]
    rows = []
    for i, r in enumerate(todo.itertuples(), 1):
        work = to_work_res(np.array(Image.open(r.image_path).convert("RGB")))
        gt = cv2.resize((np.array(Image.open(r.gt_path)) > 127).astype(np.uint8),
                        (work.shape[1], work.shape[0]), interpolation=cv2.INTER_NEAREST)
        k = 2 * max(3, int(0.03 * max(gt.shape))) + 1           # ring ~3% of the image wide
        ring = (cv2.dilate(gt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0) & (gt == 0)
        lab = cv2.cvtColor(work, cv2.COLOR_RGB2LAB).astype(float)
        les = gt > 0
        rec = {"isic_id": r.isic_id}
        if les.sum() > 50 and ring.sum() > 50:
            L = lab[..., 0]
            pix = lab[les]
            rec.update(lesion_L=float(np.median(L[les])), skin_L=float(np.median(L[ring])),
                       contrast=float(np.median(L[ring]) - np.median(L[les])),
                       heterogeneity=float(L[les].std()),
                       colour_heterogeneity=float(np.linalg.norm(pix - pix.mean(0), axis=1).mean()))
        rows.append(rec)
        if i % 200 == 0:
            print(f"    lesion characteristics {i}/{len(todo)}")
    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(path, index=False)
    return out


def contrast_heterogeneity(per_image: pd.DataFrame, chars: pd.DataFrame) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """B10 — Does Otsu win on dark lesions and SAM lose on heterogeneous ones?
    Images are split into thirds (low / medium / high) of contrast and of heterogeneity.
    (a) per cell of the 3 x 3 grid: n, median IoU and failure rate of each method, and the
        paired difference first method - second method (Wilcoxon, Holm over cells);
    (b) Spearman correlations of each method's IoU, and of the paired difference, with contrast,
        heterogeneity and colour heterogeneity. If SAM loses to Otsu as lesions become more
        heterogeneous, the difference should correlate positively with heterogeneity."""
    iou = per_image.pivot(index="isic_id", columns="method", values="iou")
    d = chars.dropna(subset=["contrast", "heterogeneity"]).set_index("isic_id").join(iou, how="inner")
    d["contrast_group"] = pd.qcut(d.contrast, 3, labels=TERTILES)
    d["heterogeneity_group"] = pd.qcut(d.heterogeneity, 3, labels=TERTILES)
    methods = [m for m in METHODS if m in iou.columns]
    a, b = (methods + [None])[:2] if len(methods) >= 2 else (methods[0], None)
    if {"color_v2", "sam_v2"} <= set(methods):
        a, b = "color_v2", "sam_v2"

    cells = []
    for (cg, hg), g in d.groupby(["contrast_group", "heterogeneity_group"], observed=True):
        rec = {"contrast": cg, "heterogeneity": hg, "n_images": len(g),
               "contrast_range": f"{g.contrast.min():.0f} to {g.contrast.max():.0f}",
               "heterogeneity_range": f"{g.heterogeneity.min():.1f} to {g.heterogeneity.max():.1f}"}
        for m in methods:
            rec[f"median_iou_{METHODS[m]}"] = g[m].median()
            rec[f"failure_rate_{METHODS[m]}"] = (g[m] < FAIL_IOU).mean()
        if b:
            diff = g[a] - g[b]
            rec[f"mean_diff_{METHODS[a]}_minus_{METHODS[b]}"] = diff.mean()
            rec[f"share_{METHODS[a]}_better"] = (diff > 0).mean()
            rec["p_wilcoxon"] = stats.wilcoxon(g[a], g[b]).pvalue if (diff != 0).any() else np.nan
        cells.append(rec)
    cells = pd.DataFrame(cells)
    if "p_wilcoxon" in cells:
        cells["p_holm"] = holm(cells.p_wilcoxon.tolist())

    corr = []
    targets = {METHODS[m]: d[m] for m in methods}
    if b:
        targets[f"{METHODS[a]} - {METHODS[b]}"] = d[a] - d[b]
    for name, y in targets.items():
        for x in ("contrast", "heterogeneity", "colour_heterogeneity"):
            rho, p = stats.spearmanr(d[x], y, nan_policy="omit")
            corr.append({"iou_of": name, "versus": x, "spearman_rho": rho, "p_value": p, "n": int(y.notna().sum())})
    corr = pd.DataFrame(corr)
    corr["p_holm"] = holm(corr.p_value.tolist())
    heatmaps(cells, methods, a, b)
    return cells, corr


def heatmaps(cells: pd.DataFrame, methods: list, a: str, b: "str | None") -> None:
    """Median IoU per contrast x heterogeneity cell for each method (+ the mean paired difference)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [(f"median_iou_{METHODS[m]}", METHODS[m], "viridis", (0.3, 0.9)) for m in methods]
    if b:
        panels.append((f"mean_diff_{METHODS[a]}_minus_{METHODS[b]}", f"{METHODS[a]} − {METHODS[b]} (mean IoU)", "RdBu", (-0.1, 0.1)))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 3.9))
    axes = np.atleast_1d(axes)
    for ax, (col, title, cmap, (vmin, vmax)) in zip(axes, panels):
        grid = cells.pivot(index="contrast", columns="heterogeneity", values=col).reindex(index=TERTILES[::-1], columns=TERTILES)
        im = ax.imshow(grid.to_numpy(dtype=float), cmap=cmap, vmin=vmin, vmax=vmax)
        for i in range(3):
            for j in range(3):
                v = grid.to_numpy(dtype=float)[i, j]
                ax.text(j, i, f"{v:.2f}" if np.isfinite(v) else "", ha="center", va="center", fontsize=10,
                        color="white" if cmap == "viridis" and v < 0.6 else "black")
        ax.set_xticks(range(3), TERTILES)
        ax.set_yticks(range(3), TERTILES[::-1])
        ax.set_xlabel("heterogeneity")
        ax.set_ylabel("contrast (lesion darker than skin)")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("IoU by lesion contrast and heterogeneity (thirds of the images)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "b10_contrast_heterogeneity.png", dpi=130)
    plt.close(fig)


def run_b10(pairs: pd.DataFrame, per_image: pd.DataFrame) -> None:
    print("\nB10 contrast / heterogeneity")
    pairs = pairs[pairs.isic_id.isin(per_image.isic_id)]
    chars = lesion_characteristics(pairs)
    cells, corr = contrast_heterogeneity(per_image, chars)
    cells.to_csv(OUT / "b10_contrast_heterogeneity_cells.csv", index=False)
    corr.to_csv(OUT / "b10_contrast_heterogeneity_correlations.csv", index=False)
    print(cells.round(3).to_string(index=False))
    print()
    print(corr.round(4).to_string(index=False))


def write_summary_excel() -> None:
    """All result tables in one workbook (benchmark_summary.xlsx), built from the saved CSVs."""
    per_image = pd.read_csv(OUT / "per_image_masks.csv")
    per_image["method"] = per_image["method"].astype(str)
    bins, below = iou_distribution(per_image)
    sheets = {"accuracy": "b1_accuracy_summary.csv", "pairwise": "b2_pairwise_accuracy.csv",
              "artifact_subgroups": "b3_artifact_subgroups.csv", "mask_agreement": "b4_mask_agreement_summary.csv",
              "contrast_heterogeneity": "b10_contrast_heterogeneity_cells.csv",
              "contrast_heterog_corr": "b10_contrast_heterogeneity_correlations.csv"}
    with pd.ExcelWriter(OUT / "benchmark_summary.xlsx") as xw:
        for name, csv in sheets.items():
            if (OUT / csv).exists():
                pd.read_csv(OUT / csv).to_excel(xw, sheet_name=name, index=False)
        bins.to_excel(xw, sheet_name="iou_bins", index=False)
        below.to_excel(xw, sheet_name="iou_cutoffs", index=False)
        failure_overlap(per_image).to_excel(xw, sheet_name="failure_overlap", index=False)
    print(f"Saved: {OUT / 'benchmark_summary.xlsx'}")
    print(below.to_string(index=False))


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), default="train",
                    help="official ISIC 2018 split; outputs go to isic2018_benchmark/train/ or isic2018_benchmark/test/")
    ap.add_argument("--images", type=Path, default=None, help="override the split's image folder")
    ap.add_argument("--gt", type=Path, default=None, help="override the split's ground-truth folder")
    ap.add_argument("--methods", nargs="+", default=[m for m in METHODS if m not in TRAINED_ON_ISIC_TRAIN],
                    choices=list(METHODS))
    ap.add_argument("--n", type=int, default=None, help="random subset of images (default: all)")
    ap.add_argument("--n-montage", type=int, default=8)
    ap.add_argument("--n-qc", type=int, default=8, help="random images in the QC montage (B7)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary-only", action="store_true",
                    help="only rebuild benchmark_summary.xlsx from the saved CSVs (no segmentation)")
    ap.add_argument("--analysis-only", action="store_true",
                    help="no segmentation: run B10 (contrast / heterogeneity) on the saved per_image_masks.csv, "
                         "then rebuild the summary workbook")
    args = ap.parse_args()
    images, gt, OUT = SPLITS[args.split]
    args.images, args.gt = args.images or images, args.gt or gt
    if args.split == "train" and TRAINED_ON_ISIC_TRAIN & set(args.methods):
        raise SystemExit(f"{sorted(TRAINED_ON_ISIC_TRAIN & set(args.methods))} was trained on the training "
                         f"split, so it can only be scored with --split test.")
    print(f"Split: {args.split} | methods: {args.methods} | outputs: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        write_summary_excel()
        return
    if args.analysis_only:
        run_b10(find_pairs(args.images, args.gt), pd.read_csv(OUT / "per_image_masks.csv"))
        write_summary_excel()
        return
    rng = np.random.default_rng(args.seed)

    pairs = find_pairs(args.images, args.gt)
    if args.n and args.n < len(pairs):
        pairs = pairs.sample(n=args.n, random_state=args.seed).sort_values("isic_id").reset_index(drop=True)
        print(f"Random subset: {len(pairs)} images (seed {args.seed})")

    print("Preprocessing stats (hair, field of view)...")
    pre = preprocessing_stats(pairs)

    print("Segmenting / loading masks and scoring...")
    seg_cache, rows, smalls = {}, [], {m: {} for m in args.methods}
    for i, r in enumerate(pairs.itertuples(), 1):
        image = np.array(Image.open(r.image_path).convert("RGB"))
        gt = np.array(Image.open(r.gt_path)) > 127
        for m in args.methods:
            pred, info = predicted_mask(m, r, image, seg_cache)
            if pred.shape != gt.shape:
                pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            rows.append({"isic_id": r.isic_id, "method": m, **metrics(pred, gt),
                         **{f"qc_{k}": v for k, v in info.items()}})
            smalls[m][r.isic_id] = small(pred, SMALL)
        if i % 50 == 0:
            print(f"  {i}/{len(pairs)}")
    per_image = pd.DataFrame(rows).merge(pre, on="isic_id", how="left")
    per_image.to_csv(OUT / "per_image_masks.csv", index=False)

    print("\nB1 accuracy"); b1 = accuracy_summary(per_image); b1.to_csv(OUT / "b1_accuracy_summary.csv", index=False)
    print(b1[["method", "n_images", "iou_mean", "iou_median", "dice_mean", "thresholded_jaccard_mean",
              "failure_rate", "sensitivity_mean", "specificity_mean", "area_ratio_median"]].round(3).to_string(index=False))
    iou_plot(per_image)

    print("\nB2 pairwise accuracy"); b2 = pairwise_accuracy(per_image, args.n_boot, rng); b2.to_csv(OUT / "b2_pairwise_accuracy.csv", index=False)
    print(b2[["method_a", "method_b", "mean_iou_difference_a_minus_b", "diff_ci95_low", "diff_ci95_high",
              "p_wilcoxon_holm", "fails_only_a", "fails_only_b", "p_mcnemar_holm"]].round(4).to_string(index=False))

    print("\nB3 artifact subgroups"); b3 = artifact_subgroups(per_image, pre); b3.to_csv(OUT / "b3_artifact_subgroups.csv", index=False)
    print(b3.round(3).to_string(index=False))

    print("\nB4 mask agreement"); b4i, b4 = mask_agreement(pairs, smalls)
    b4i.to_csv(OUT / "b4_mask_agreement_images.csv", index=False); b4.to_csv(OUT / "b4_mask_agreement_summary.csv", index=False)
    print(b4.round(3).to_string(index=False))

    print("\nB5 disagreement montages / B6 failure montages / B7 QC montage")
    disagreement_montages(pairs, b4i, per_image, args.n_montage)
    failure_montages(pairs, per_image, args.n_montage)
    qc_montage(pairs, per_image, args.methods, args.n_qc, args.seed)
    run_b10(pairs, per_image)
    print("\nB8/B9 IoU distribution + failure overlap -> summary workbook")
    write_summary_excel()
    print(f"\nAll outputs: {OUT}")


if __name__ == "__main__":
    main()
