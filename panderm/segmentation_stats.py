"""
Segmentation statistics — compares every segmentation method run so far
=========================================================================
Methods: color (Otsu v1), color_v2 (Otsu v2), sam (SAM v1), sam_v1_pre
(SAM v1 + v2 preprocessing), sam_v2 (Otsu-prompted SAM).

Questions answered (one function each, see their docstrings):
    Q1  failure_rates()          How often does each method fail / fall back?
    Q2  failure_by_class()       Is the failure rate different for DN / MIA / in situ?
    Q3  artifacts_vs_errors()    Are patients with vignette / heavy hair / a fallback
                                 misclassified more often?
    Q4  mask_agreement()         How similar are two methods' masks on the same image?
    Q5  mask_size_differences()  Does one method systematically give larger masks?
    Q6  disagreement_montages()  Where do two methods disagree?

Masks are NOT re-segmented when a cached copy exists: the mask-guided runs
already saved full-resolution masks in features/maskguided/<backend>/masks/.
Methods without cached masks (color and sam: their runs kept only crops,
which cannot be mapped back to image positions) are segmented ONCE and cached
in the same place, so later runs — and mask-guided pooling — reuse them.

Unit of analysis: images of one patient are not independent, so tests are
run at PATIENT level (a patient "has a failure" if any of its images does;
continuous values = the patient's median). Image-level numbers are reported
alongside for description only.

Failure definitions (FAILURE_RULE below):
    color       fell back to the full frame (mask < 1% -> whole image)
    color_v2    fell back to the field of view      (QC file: fallback)
    sam         implausible mask: < 3% or > 80% of the frame (v1 has no fallback)
    sam_v1_pre  implausible mask: < 3% or > 80% of the frame (no fallback)
    sam_v2      SAM mask rejected -> Otsu v2 / field of view (QC file: fallback)
"Implausible" (< 3% or > 80% of the frame) is also reported for EVERY method
as one common yardstick.

Outputs: OUTPUT_DIR/stats_segmentation/
    q1_failure_rates.csv, q2_failure_by_class.csv, q3_artifacts_vs_errors.csv,
    q4_mask_agreement_images.csv, q4_mask_agreement_summary.csv,
    q5_mask_area_tests.csv, q5_bland_altman.png,
    q6_disagreement_<a>_vs_<b>.png + .csv, per_image_masks.csv

Q3 needs results/<mode>/<backend>/patient_predictions_<mode>.csv, written by
evaluate.py — re-run evaluate.py for any result folder that lacks it.

Usage:
    python panderm/segmentation_stats.py
    python panderm/segmentation_stats.py --n-perm 20000 --n-montage 10
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
from config import OUTPUT_DIR, DATA_ROOT, CLASS_NAMES

METHODS = {"color": "Otsu v1", "color_v2": "Otsu v2", "sam": "SAM v1",
           "sam_v1_pre": "SAM v1 + preproc", "sam_v2": "SAM v2"}
FAILURE_RULE = {"color": "full-frame fallback", "color_v2": "fallback to field of view",
                "sam": "implausible (<3% or >80% of frame)",
                "sam_v1_pre": "implausible (<3% or >80% of frame)",
                "sam_v2": "SAM mask rejected (fallback)"}
TINY, WHOLE = 0.03, 0.80
SMALL = 256            # long side for pairwise mask comparisons (area fractions are unaffected)
OUT = OUTPUT_DIR / "stats_segmentation"
MASK_ROOT = OUTPUT_DIR / "features" / "maskguided"
DEFAULT_PAIRS = [("color_v2", "sam_v2"), ("sam_v1_pre", "sam_v2"), ("color_v2", "sam_v1_pre"),
                 ("color", "color_v2"), ("sam", "sam_v1_pre")]


# ── Small statistics helpers ─────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.96) -> "tuple[float, float]":
    """95% Wilson score interval for a proportion k/n (better than p ± 1.96·SE
    for small counts or rates near 0 or 1)."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def holm(p: "list[float]") -> "list[float]":
    """Holm-Bonferroni adjusted p-values (NaNs left as NaN)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    vals = p[ok][idx] * (ok.sum() - np.arange(ok.sum()))
    vals = np.minimum(np.maximum.accumulate(vals), 1.0)
    tmp = np.empty_like(vals)
    tmp[idx] = vals
    out[ok] = tmp
    return out.tolist()


def chi2_stat(cls: np.ndarray, fail: np.ndarray, k: int) -> float:
    """Pearson chi-square of the (class x failed) table, no continuity correction."""
    obs = np.stack([np.bincount(cls[~fail], minlength=k), np.bincount(cls[fail], minlength=k)], axis=1)
    exp = obs.sum(1, keepdims=True) * obs.sum(0, keepdims=True) / obs.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nansum((obs - exp) ** 2 / exp))


def iou_dice(a: np.ndarray, b: np.ndarray) -> "tuple[float, float]":
    inter, union, total = (a & b).sum(), (a | b).sum(), a.sum() + b.sum()
    if union == 0:            # both empty: they agree
        return 1.0, 1.0
    return inter / union, 2 * inter / total


# ── Masks: cached, segmented only if missing ─────────────────────────────

def mask_file(backend: str, image_path: str) -> Path:
    """Same layout extract_features_maskguided.py uses for its mask cache."""
    return (MASK_ROOT / backend / "masks" / Path(image_path).relative_to(DATA_ROOT)).with_suffix(".png")


def load_masks(manifest: pd.DataFrame, backend: str) -> "tuple[list, pd.DataFrame]":
    """Returns (small masks in manifest order, per-image mask stats). Loads the
    cached full-resolution mask; if absent, segments ONCE and caches it."""
    seg, small, rows, n_new = None, [], [], 0
    for _, r in manifest.iterrows():
        p = mask_file(backend, r["image_path"])
        if p.exists():
            m = np.array(Image.open(p)) > 0
        else:
            if seg is None:
                from segment_lesions import build_segmenter
                print(f"  segmenting missing '{backend}' masks once (cached afterwards)...")
                seg = build_segmenter(backend)
            m = seg.segment(np.array(Image.open(r["image_path"]).convert("RGB"))) > 0
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(m.astype(np.uint8) * 255).save(p)
            n_new += 1
            if n_new % 50 == 0:
                print(f"    {n_new} segmented")
        h, w = m.shape
        s = SMALL / max(h, w)
        small.append(cv2.resize(m.astype(np.uint8), (round(w * s), round(h * s)),
                                interpolation=cv2.INTER_NEAREST) > 0)
        frac = float(m.mean())
        rows.append({"image_path": r["image_path"], "method": backend, "mask_frac_of_frame": frac,
                     "full_frame": frac >= 0.999, "tiny": frac < TINY, "whole": frac > WHOLE})
    print(f"  {backend:<11} {len(small)} masks ({n_new} newly segmented)")
    return small, pd.DataFrame(rows)


def add_failure_flags(per_image: pd.DataFrame) -> pd.DataFrame:
    """Method-specific failure (FAILURE_RULE) + the common 'implausible' flag.
    v2 methods read their own fallback decision from the segment stage's QC file."""
    per_image["implausible"] = per_image["tiny"] | per_image["whole"]
    per_image["failure"] = per_image["implausible"]
    per_image.loc[per_image.method == "color", "failure"] = per_image["full_frame"]
    for b in ("color_v2", "sam_v2"):
        qc = OUTPUT_DIR / "segmented_cache" / b / f"segmentation_qc_{b}.csv"
        if qc.exists():
            fb = pd.read_csv(qc).set_index("image_path")["fallback"].astype(bool)
            sel = per_image.method == b
            per_image.loc[sel, "failure"] = per_image.loc[sel, "image_path"].map(fb).fillna(False).values
        else:
            print(f"  WARNING: {qc.name} not found — '{b}' failure falls back to 'implausible'")
    return per_image


# ── Q1 ───────────────────────────────────────────────────────────────────

def failure_rates(per_image: pd.DataFrame) -> pd.DataFrame:
    """Q1 — How often does each method fail / fall back?
    Failure rate with a 95% Wilson interval, overall and per class, at image
    level and patient level (patient = any of its images failed), for the
    method's own failure rule and for the common 'implausible' yardstick."""
    rows = []
    for m, g in per_image.groupby("method", sort=False):
        pat = g.groupby("group_id").agg(diagnosis=("diagnosis", "first"),
                                        failure=("failure", "any"), implausible=("implausible", "any"))
        for scope in ["all"] + CLASS_NAMES:
            gi = g if scope == "all" else g[g.diagnosis == scope]
            gp = pat if scope == "all" else pat[pat.diagnosis == scope]
            for flag in ("failure", "implausible"):
                for level, d in (("image", gi), ("patient", gp)):
                    k, n = int(d[flag].sum()), len(d)
                    lo, hi = wilson_ci(k, n)
                    rows.append({"method": METHODS[m], "rule": FAILURE_RULE[m] if flag == "failure" else "implausible (<3% or >80%)",
                                 "flag": flag, "level": level, "class": scope, "n": n, "failures": k,
                                 "rate": k / n if n else np.nan, "ci95_low": lo, "ci95_high": hi})
    return pd.DataFrame(rows)


# ── Q2 ───────────────────────────────────────────────────────────────────

def failure_by_class(per_image: pd.DataFrame, n_perm: int, rng) -> pd.DataFrame:
    """Q2 — Is the failure rate different for DN / MIA / in situ?
    Chi-square test of (class x failed), p-value by permutation (valid when
    expected counts are small, where the chi-square approximation is not),
    effect size Cramér's V. Primary: patient level; image level for description."""
    rows = []
    for m, g in per_image.groupby("method", sort=False):
        pat = g.groupby("group_id").agg(diagnosis=("diagnosis", "first"), failure=("failure", "any"))
        for level, d in (("patient", pat), ("image", g)):
            cls = d["diagnosis"].map({c: i for i, c in enumerate(CLASS_NAMES)}).to_numpy()
            fail = d["failure"].to_numpy(bool)
            obs = chi2_stat(cls, fail, len(CLASS_NAMES))
            if fail.sum() in (0, len(fail)):
                p, v = np.nan, np.nan
            else:
                perm = np.array([chi2_stat(cls, rng.permutation(fail), len(CLASS_NAMES)) for _ in range(n_perm)])
                p = (1 + (perm >= obs - 1e-12).sum()) / (n_perm + 1)
                v = np.sqrt(obs / len(fail))
            rates = {f"rate_{c}": fail[cls == i].mean() for i, c in enumerate(CLASS_NAMES)}
            rows.append({"method": METHODS[m], "rule": FAILURE_RULE[m], "level": level, "n": len(fail),
                         "failures": int(fail.sum()), **rates, "chi2": obs, "p_permutation": p, "cramers_v": v})
    out = pd.DataFrame(rows)
    sel = out.level == "patient"
    out.loc[sel, "p_holm"] = holm(out.loc[sel, "p_permutation"].tolist())
    return out


# ── Q3 ───────────────────────────────────────────────────────────────────

def classifier_runs() -> "list[tuple[str, Path, str | None]]":
    """(label, patient_predictions csv, backend whose fallback applies) for every
    evaluated classifier; runs without a predictions file are listed as missing."""
    runs = [("Frozen CLS, raw", OUTPUT_DIR / "results/frozen/none/patient_predictions_frozen.csv", None),
            ("Partial fine-tune, raw", OUTPUT_DIR / "results/finetune/none/patient_predictions_finetune.csv", None)]
    for b in METHODS:
        runs.append((f"Frozen CLS, {METHODS[b]} crop",
                     OUTPUT_DIR / f"results/frozen/{b}/patient_predictions_frozen.csv", b))
        runs.append((f"Mask-guided, {METHODS[b]} masks",
                     OUTPUT_DIR / f"results/maskguided/{b}/patient_predictions_maskguided.csv", b))
    return runs


def artifacts_vs_errors(per_image: pd.DataFrame, n_perm: int, rng) -> pd.DataFrame:
    """Q3 — Are patients with vignette / heavy hair / a fallback misclassified more often?
    Patient-level artifacts (from the shared v2 preprocessing QC, color_v2 file):
      vignette    any image had a detected vignette (field of view < 99% of frame)
      heavy hair  patient's median hair fraction in the top quartile of patients
      fallback    any image failed under the method that produced this classifier's input
    For each classifier: error rate with vs without the artifact, Fisher's exact
    test (odds ratio), and a class-stratified permutation test (artifact labels
    shuffled WITHIN each class) so a class that is both harder and more
    artifact-prone cannot create the association on its own. Holm over all tests."""
    qc = pd.read_csv(OUTPUT_DIR / "segmented_cache" / "color_v2" / "segmentation_qc_color_v2.csv")
    qc["group_id"] = qc["patient_id"].astype(str) + "__" + qc["diagnosis"]
    pat = qc.groupby("group_id").agg(vignette=("fov_frac", lambda s: (s < 0.99).any()),
                                     hair=("hair_frac", "median"))
    pat["heavy_hair"] = pat["hair"] >= pat["hair"].quantile(0.75)
    fails = per_image.groupby(["method", "group_id"])["failure"].any()

    rows = []
    for label, path, backend in classifier_runs():
        if not path.exists():
            print(f"  Q3: no predictions for '{label}' — re-run evaluate.py for {path.parent}")
            continue
        pr = pd.read_csv(path).set_index("group_id")
        d = pat.join(pr[["true_label", "pred_label"]], how="inner")
        d["error"] = d["true_label"] != d["pred_label"]
        artifacts = {"vignette": d["vignette"].astype(bool), "heavy_hair": d["heavy_hair"].astype(bool)}
        if backend is not None:
            artifacts["fallback"] = d.index.map(fails.loc[backend]).fillna(False).astype(bool)
        cls = d["true_label"].to_numpy()
        err = d["error"].to_numpy(bool)
        for name, art in artifacts.items():
            a = np.asarray(art, bool)
            if a.sum() in (0, len(a)):
                continue
            table = [[(a & err).sum(), (a & ~err).sum()], [(~a & err).sum(), (~a & ~err).sum()]]
            odds, p_fisher = stats.fisher_exact(table)
            diff = err[a].mean() - err[~a].mean()
            null = []
            for _ in range(n_perm):
                ap = a.copy()
                for c in np.unique(cls):
                    idx = np.where(cls == c)[0]
                    ap[idx] = rng.permutation(ap[idx])
                null.append(err[ap].mean() - err[~ap].mean())
            p_strat = (1 + (np.abs(null) >= abs(diff) - 1e-12).sum()) / (n_perm + 1)
            rows.append({"classifier": label, "artifact": name, "n_patients": len(a),
                         "n_with_artifact": int(a.sum()), "error_rate_with": err[a].mean(),
                         "error_rate_without": err[~a].mean(), "difference": diff,
                         "odds_ratio": odds, "p_fisher": p_fisher, "p_class_stratified": p_strat})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p_class_stratified"].tolist())
    return out


# ── Q4 ───────────────────────────────────────────────────────────────────

def mask_agreement(manifest: pd.DataFrame, masks: dict) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Q4 — How similar are two methods' masks on the same image?
    IoU (overlap / union) and Dice (2·overlap / total) for every pair of
    methods on every image; summary = median and IQR, overall and per class."""
    rows = []
    for a, b in combinations(masks, 2):
        for i, r in manifest.iterrows():
            iou, dice = iou_dice(masks[a][i], masks[b][i])
            rows.append({"image_path": r["image_path"], "diagnosis": r["diagnosis"],
                         "method_a": METHODS[a], "method_b": METHODS[b], "iou": iou, "dice": dice})
    img = pd.DataFrame(rows)
    summ = []
    for (a, b), g in img.groupby(["method_a", "method_b"], sort=False):
        for scope in ["all"] + CLASS_NAMES:
            d = g if scope == "all" else g[g.diagnosis == scope]
            summ.append({"method_a": a, "method_b": b, "class": scope, "n_images": len(d),
                         "iou_median": d.iou.median(), "iou_q25": d.iou.quantile(.25), "iou_q75": d.iou.quantile(.75),
                         "dice_median": d.dice.median(), "share_iou_below_0.5": (d.iou < 0.5).mean()})
    return img, pd.DataFrame(summ)


# ── Q5 ───────────────────────────────────────────────────────────────────

def mask_size_differences(per_image: pd.DataFrame) -> pd.DataFrame:
    """Q5 — Does one method systematically give larger masks?
    Per patient: median mask area (share of the frame). For each pair of methods,
    Wilcoxon signed-rank test on the paired patient areas (no normality assumption),
    median difference, and the share of patients where method A is larger. Holm-corrected.
    Also draws Bland–Altman plots (difference vs mean area, with mean bias and
    ±1.96 SD limits) for every pair."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    area = per_image.pivot_table(index="group_id", columns="method", values="mask_frac_of_frame", aggfunc="median")
    pairs = list(combinations([m for m in METHODS if m in area.columns], 2))
    rows = []
    ncol = 5
    fig, axes = plt.subplots(int(np.ceil(len(pairs) / ncol)), ncol, figsize=(4 * ncol, 3.6 * np.ceil(len(pairs) / ncol)))
    axes = np.atleast_1d(axes).ravel()
    for ax, (a, b) in zip(axes, pairs):
        d = area[[a, b]].dropna()
        diff, mean = d[a] - d[b], (d[a] + d[b]) / 2
        w = stats.wilcoxon(d[a], d[b]) if (diff != 0).any() else None
        rows.append({"method_a": METHODS[a], "method_b": METHODS[b], "n_patients": len(d),
                     "median_area_a": d[a].median(), "median_area_b": d[b].median(),
                     "median_difference_a_minus_b": diff.median(), "share_a_larger": (diff > 0).mean(),
                     "wilcoxon_stat": w.statistic if w else np.nan, "p_wilcoxon": w.pvalue if w else np.nan})
        bias, sd = diff.mean(), diff.std()
        ax.scatter(mean, diff, s=10, alpha=0.6, color="0.35")
        for y, ls in ((bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")):
            ax.axhline(y, color="tab:blue", ls=ls, lw=1)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_title(f"{METHODS[a]} − {METHODS[b]}", fontsize=9)
        ax.set_xlabel("mean area (share of frame)", fontsize=8)
        ax.set_ylabel("difference", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[len(pairs):]:
        ax.axis("off")
    fig.suptitle("Bland–Altman: patient median mask area, method A − method B (bias and ±1.96 SD)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "q5_bland_altman.png", dpi=130)
    plt.close(fig)
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_wilcoxon"].tolist())
    return out


# ── Q6 ───────────────────────────────────────────────────────────────────

def disagreement_montages(manifest: pd.DataFrame, masks: dict, agreement: pd.DataFrame,
                          pairs, n: int) -> None:
    """Q6 — Where do two methods disagree?
    For each chosen pair: the n images with the LOWEST IoU, raw image with both
    mask outlines (method A green, method B magenta), saved as a montage + CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    index = {p: i for i, p in enumerate(manifest["image_path"])}
    for a, b in pairs:
        if a not in masks or b not in masks:
            continue
        g = agreement[(agreement.method_a == METHODS[a]) & (agreement.method_b == METHODS[b])]
        worst = g.nsmallest(n, "iou")
        worst.to_csv(OUT / f"q6_disagreement_{a}_vs_{b}.csv", index=False)
        cols = 4
        fig, axes = plt.subplots(int(np.ceil(len(worst) / cols)), cols, figsize=(4.2 * cols, 3.4 * np.ceil(len(worst) / cols)))
        axes = np.atleast_1d(axes).ravel()
        for ax, (_, r) in zip(axes, worst.iterrows()):
            i = index[r["image_path"]]
            ma, mb = masks[a][i], masks[b][i]
            img = np.array(Image.open(r["image_path"]).convert("RGB").resize((ma.shape[1], ma.shape[0])))
            for m, colour in ((ma, (0, 220, 0)), (mb, (255, 0, 255))):
                cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cnt, -1, colour, 2)
            ax.imshow(img)
            ax.set_title(f"{r['diagnosis']}  IoU={r['iou']:.2f}\n{Path(r['image_path']).parent.name}", fontsize=8)
            ax.axis("off")
        for ax in axes[len(worst):]:
            ax.axis("off")
        fig.suptitle(f"Lowest agreement: {METHODS[a]} (green) vs {METHODS[b]} (magenta)", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / f"q6_disagreement_{a}_vs_{b}.png", dpi=120)
        plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--n-perm", type=int, default=10000, help="permutations for Q2/Q3 p-values")
    ap.add_argument("--n-montage", type=int, default=8, help="images per disagreement montage (Q6)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    manifest = pd.read_csv(OUTPUT_DIR / "dataset_manifest.csv").reset_index(drop=True)
    manifest["group_id"] = manifest["patient_id"].astype(str) + "__" + manifest["diagnosis"]
    print(f"{len(manifest)} images, {manifest.group_id.nunique()} patients -> {OUT}\nLoading masks:")

    masks, frames = {}, []
    for b in args.methods:
        masks[b], f = load_masks(manifest, b)
        frames.append(f.assign(group_id=manifest["group_id"].values, diagnosis=manifest["diagnosis"].values))
    per_image = add_failure_flags(pd.concat(frames, ignore_index=True))
    per_image.to_csv(OUT / "per_image_masks.csv", index=False)

    print("\nQ1 failure rates"); q1 = failure_rates(per_image); q1.to_csv(OUT / "q1_failure_rates.csv", index=False)
    print(q1[(q1.flag == "failure") & (q1["class"] == "all")][["method", "rule", "level", "failures", "n", "rate", "ci95_low", "ci95_high"]].round(3).to_string(index=False))

    print("\nQ2 failure by class"); q2 = failure_by_class(per_image, args.n_perm, rng); q2.to_csv(OUT / "q2_failure_by_class.csv", index=False)
    print(q2[q2.level == "patient"].drop(columns=["rule", "level"]).round(3).to_string(index=False))

    print("\nQ3 artifacts vs errors"); q3 = artifacts_vs_errors(per_image, args.n_perm, rng); q3.to_csv(OUT / "q3_artifacts_vs_errors.csv", index=False)
    if len(q3):
        print(q3.round(3).to_string(index=False))

    print("\nQ4 mask agreement"); q4i, q4 = mask_agreement(manifest, masks)
    q4i.to_csv(OUT / "q4_mask_agreement_images.csv", index=False); q4.to_csv(OUT / "q4_mask_agreement_summary.csv", index=False)
    print(q4[q4["class"] == "all"].drop(columns="class").round(3).to_string(index=False))

    print("\nQ5 mask size"); q5 = mask_size_differences(per_image); q5.to_csv(OUT / "q5_mask_area_tests.csv", index=False)
    print(q5.round(4).to_string(index=False))

    print("\nQ6 disagreement montages")
    disagreement_montages(manifest, masks, q4i, DEFAULT_PAIRS, args.n_montage)
    print(f"\nAll outputs: {OUT}")


if __name__ == "__main__":
    main()
