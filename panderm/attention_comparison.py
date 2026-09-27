"""
Attention Concentration — none vs color vs sam
================================================
Independent evidence for the segmentation-backend comparison, on a
different axis than the classification numbers: where does frozen
PanDerm's own attention go, under each preprocessing condition?

Reuses the CLS-token attention hook from evaluate.py's get_attention_map(),
pointed at the FROZEN checkpoint (CHECKPOINT_LARGE) instead of a fine-tuned
fold checkpoint — matches what extract_features.py/evaluate.py actually use
for these three backends. Needs no SAM2/sam2env — only PanDerm (already in
this env) + the segmented_cache files already saved by --stage segment.

Metrics (all three backends, no reference mask needed):
    entropy               Shannon entropy over the patch grid. Lower = more focused.
    top10_concentration   attention mass in the top 10% of patches. Higher = more focused.
    centroid_offset       attention centroid's distance from frame centre / diagonal.
    fg_attention_fraction color/sam only — attention mass landing on non-black
                          (kept) pixels of that backend's own crop.

Statistics: paired across the 3 backends on identical images — Friedman
(omnibus) + pairwise Wilcoxon with Holm correction.

The masking-robustness experiment (is attention more fragile on low-pigment
lesions?) has moved to masking_experiment.py, which imports load_frozen_encoder,
raw_attention_map, attention_metrics, and cache_path from this file.

Usage:
    python panderm/attention_comparison.py --n 100
    python panderm/attention_comparison.py --n 100 --backends none color sam
"""

import argparse
import sys
from collections import OrderedDict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PANDERM_CLASS, CHECKPOINT_LARGE, OUTPUT_DIR,
    IMAGENET_MEAN, IMAGENET_STD, NB_CLASSES, RANDOM_SEED,
)

RES_DIR = OUTPUT_DIR / "attention_comparison"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BACKBONE_BACKENDS = ["none", "color", "sam"]   # segmenters that have real cached output


# ── Model + hook (frozen encoder, same loader as evaluate.py) ───────────

def load_frozen_encoder() -> torch.nn.Module:
    for p in [str(PANDERM_CLASS.parent), str(PANDERM_CLASS)]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from models.modeling_finetune import panderm_large_patch16_224
    model = panderm_large_patch16_224(num_classes=NB_CLASSES)
    ckpt = torch.load(str(CHECKPOINT_LARGE), map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    clean = OrderedDict((k.replace("encoder.", "").replace("module.", ""), v)
                        for k, v in state.items())
    missing, unexpected = model.load_state_dict(clean, strict=False)
    print(f"Frozen encoder loaded: missing={len(missing)} unexpected={len(unexpected)} "
         f"(missing=head, unexpected=SSL decoder — both expected)")
    model.eval()
    return model.to(DEVICE)


_NORM_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def apply_patch_mask(pil_img: Image.Image, mask_frac: float = 0.2,
                     patch_size: int = 16, seed: "int | None" = None
                     ) -> "tuple[Image.Image, np.ndarray]":
    """
    Cutout-style masking on the already-224x224 image: replaces mask_frac of
    the patch_size x patch_size patches with the image's OWN mean color (not
    black — black already means "cropped away" for color/sam, so painting
    masked patches black would confound this test with that one).

    Returns (masked_img, mask_grid) where mask_grid is a boolean (side, side)
    array (side = 224 // patch_size) aligned 1:1 with raw_attention_map's
    native_grid when patch_size matches the model's patch size (16 -> 14x14).
    """
    arr = np.array(pil_img).copy()
    h, w = arr.shape[:2]
    n_h, n_w = h // patch_size, w // patch_size
    mean_color = arr.reshape(-1, 3).mean(axis=0)
    n_mask = max(1, int(round(mask_frac * n_h * n_w)))
    rng = np.random.default_rng(seed)
    flat_idx = rng.choice(n_h * n_w, size=n_mask, replace=False)
    mask_grid = np.zeros((n_h, n_w), dtype=bool)
    for idx in flat_idx:
        r, c = divmod(int(idx), n_w)
        mask_grid[r, c] = True
        arr[r * patch_size:(r + 1) * patch_size, c * patch_size:(c + 1) * patch_size] = mean_color
    return Image.fromarray(arr.astype(np.uint8)), mask_grid


def raw_attention_map(model: torch.nn.Module, img_path, mask_frac: float = 0.0,
                      patch_size: int = 16, seed: "int | None" = None
                      ) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """
    Returns (native_grid, attn_map_224, resized_rgb, mask_grid):
      native_grid   the raw (side, side) patch-grid attention (e.g. 14x14 =
                    196 bins) — this is what entropy/concentration should be
                    computed on, since it has real dynamic range.
      attn_map_224  the same thing cubic-upsampled to 224x224 and renormalised
                    — for the centroid metric and for visualization ONLY.
                    Entropy on this upsampled version is near-meaningless: at
                    50,176 bins even a sharply peaked map barely moves away
                    from the theoretical max, regardless of how concentrated
                    it visually looks.
      resized_rgb   the 224x224 RGB image the map was computed on (post-mask,
                    if mask_frac > 0).
      mask_grid     boolean (side, side) array of which native-grid patches
                    were masked, or None if mask_frac == 0.
    """
    pil_img = Image.open(img_path).convert("RGB")
    # Resize(256) + CenterCrop(224), matching extract_features.py's eval_tf exactly —
    # a plain resize((224, 224)) would squash aspect ratio instead of cropping, which
    # matters for backend="none" where img_path is the raw, uncropped image.
    resized_pil = transforms.CenterCrop(224)(transforms.Resize(256)(pil_img))
    mask_grid = None
    if mask_frac > 0:
        resized_pil, mask_grid = apply_patch_mask(resized_pil, mask_frac, patch_size, seed)
    img_tensor = _NORM_TF(resized_pil).unsqueeze(0)
    captured = {}

    def hook_fn(module, inp, out):
        x = inp[0]
        B, N, C = x.shape
        qkv = module.qkv(x).reshape(
            B, N, 3, module.num_heads, C // module.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * module.scale
        captured["weights"] = attn.softmax(dim=-1).detach().cpu()

    handle = model.blocks[-1].attn.register_forward_hook(hook_fn)
    with torch.no_grad():
        model.forward_features(img_tensor.to(DEVICE), is_train=False)
    handle.remove()

    cls_attn  = captured["weights"][0, :, 0, 1:]      # (heads, n_patches)
    mean_attn = cls_attn.mean(dim=0)                  # average over heads
    side      = int(np.sqrt(mean_attn.shape[0]))
    native_grid = mean_attn.reshape(side, side).numpy()
    native_grid = np.clip(native_grid, 0, None)
    native_grid = native_grid / (native_grid.sum() + 1e-9)

    attn_map_224 = cv2.resize(native_grid, (224, 224), interpolation=cv2.INTER_CUBIC)
    attn_map_224 = np.clip(attn_map_224, 0, None)
    attn_map_224 = attn_map_224 / (attn_map_224.sum() + 1e-9)

    resized = np.array(resized_pil)
    return native_grid, attn_map_224, resized, mask_grid


# ── Metrics ──────────────────────────────────────────────────────────────

def attention_metrics(native_grid: np.ndarray, attn_map_224: np.ndarray,
                      resized_rgb: "np.ndarray | None" = None,
                      mask_grid: "np.ndarray | None" = None) -> dict:
    # entropy / concentration on the NATIVE patch grid (real dynamic range) —
    # NOT the 224x224 upsampled map, which is too many bins to be sensitive.
    p = native_grid.flatten()
    p = p / (p.sum() + 1e-9)
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    max_entropy = float(np.log(p.size))   # for reporting entropy as a fraction of max
    k = max(1, int(0.10 * p.size))
    top10 = float(np.sort(p)[-k:].sum())

    # centroid on the upsampled map (finer spatial resolution for this one)
    h, w = attn_map_224.shape
    ys, xs = np.indices((h, w))
    cy = float((attn_map_224 * ys).sum())
    cx = float((attn_map_224 * xs).sum())
    diag = np.sqrt(h ** 2 + w ** 2)
    centroid_offset = float(np.sqrt((cy - h / 2) ** 2 + (cx - w / 2) ** 2) / diag)

    rec = dict(entropy=entropy, entropy_frac_of_max=entropy / max_entropy,
              top10_concentration=top10, centroid_offset=centroid_offset)

    if resized_rgb is not None:
        fg = (resized_rgb.max(axis=-1) > 8)   # non-near-black = kept foreground
        rec["fg_attention_fraction"] = float(attn_map_224[fg].sum()) if fg.any() else np.nan

    if mask_grid is not None:
        # how much attention lands ON the masked-out patches themselves —
        # ~= mask_frac_actual means attention is roughly indifferent to
        # masking; higher means attention is drawn TO the masked squares
        # (the same kind of artifact-seeking failure as the background-bias
        # bug); lower means the model correctly reallocates to real content.
        rec["attention_on_masked"] = float(native_grid[mask_grid].sum())
        rec["mask_frac_actual"] = float(mask_grid.mean())
    return rec


def cache_path(row: pd.Series, backend: str) -> Path:
    if backend == "none":
        return Path(row["image_path"])
    img_name = Path(row["image_path"]).name
    folder = row.get("folder_name") or row.get("patient_id")
    return OUTPUT_DIR / "segmented_cache" / backend / row["diagnosis"] / str(folder) / img_name


def pigment_score(row: pd.Series, backend: str = "color") -> float:
    """Mean HSV saturation within the crop's foreground (backend != 'none')
    or the whole resized frame (backend == 'none'). Low saturation ~ a
    low-pigment lesion. Objective stand-in for eyeballing "less pigment".
    Shared by masking_experiment.py — always score off "color" there too,
    since scoring off the raw frame picks by whole-image color, not lesion
    pigment."""
    p = cache_path(row, backend)
    if not p.exists():
        return np.nan
    img = np.array(Image.open(p).convert("RGB").resize((224, 224)))
    fg = img.max(axis=-1) > 8 if backend != "none" else np.ones(img.shape[:2], dtype=bool)
    if not fg.any():
        return np.nan
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return float(hsv[..., 1][fg].mean())


def sample_images(n: int, seed: int) -> pd.DataFrame:
    manifest = pd.read_csv(OUTPUT_DIR / "dataset_manifest.csv")
    n = min(n, len(manifest))
    per_class = max(1, n // manifest["diagnosis"].nunique())
    parts = [g.sample(n=min(per_class, len(g)), random_state=seed)
             for _, g in manifest.groupby("diagnosis")]
    sample = pd.concat(parts, ignore_index=True)
    if len(sample) > n:
        sample = sample.sample(n=n, random_state=seed)
    return sample.reset_index(drop=True)


# ── Statistics ───────────────────────────────────────────────────────────

def holm_correct(p_raw: "list[float]") -> "list[float]":
    """Holm-Bonferroni across a family of p-values."""
    order = np.argsort(p_raw)
    m = len(p_raw)
    holm = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p_raw[idx]
        running_max = max(running_max, adj)
        holm[idx] = min(running_max, 1.0)
    return holm.tolist()


def friedman_and_posthoc(df: pd.DataFrame, metric: str) -> "tuple[dict, pd.DataFrame]":
    wide = df.pivot(index="image_path", columns="backend", values=metric).dropna()
    backends = [b for b in BACKBONE_BACKENDS if b in wide.columns]
    wide = wide[backends]

    stat, p = stats.friedmanchisquare(*[wide[b] for b in backends])
    omnibus = {"metric": metric, "n_images": len(wide), "friedman_stat": stat, "friedman_p": p}

    pairs = list(combinations(backends, 2))
    rows = []
    for a, b in pairs:
        w, wp = stats.wilcoxon(wide[a], wide[b])
        rows.append({"metric": metric, "a": a, "b": b, "n": len(wide),
                    "delta_a_minus_b": float((wide[a] - wide[b]).mean()),
                    "wilcoxon_stat": w, "p_raw": wp})
    posthoc = pd.DataFrame(rows)
    if not posthoc.empty:
        posthoc["p_holm"] = holm_correct(posthoc["p_raw"].tolist())
    return omnibus, posthoc


def pigment_group_test(df: pd.DataFrame, metric_cols: list) -> pd.DataFrame:
    """Median-splits the sampled images into low-/high-pigment groups (scored
    off the "color" crop) and runs a Mann-Whitney U test per backend per
    metric — an actual significance test for the "low-pigment lesions get
    worse/more diffuse attention" claim, rather than eyeballing overlays.
    Independent-groups test (unpaired), since a low- and a high-pigment image
    are different lesions, not two views of the same one. Holm-corrected
    across each backend's metric family."""
    med = df["pigment_score"].median()
    df = df.assign(pigment_group=np.where(df["pigment_score"] < med, "low", "high"))

    rows = []
    for backend in df["backend"].unique():
        sub = df[df["backend"] == backend]
        p_raw = []
        pending = []
        for metric in metric_cols:
            low = sub.loc[sub["pigment_group"] == "low", metric].dropna()
            high = sub.loc[sub["pigment_group"] == "high", metric].dropna()
            if len(low) < 2 or len(high) < 2:
                continue
            u, wp = stats.mannwhitneyu(low, high, alternative="two-sided")
            pending.append({"backend": backend, "metric": metric,
                            "low_n": len(low), "high_n": len(high),
                            "low_median": float(low.median()), "high_median": float(high.median()),
                            "delta_low_minus_high": float(low.median() - high.median()),
                            "p_raw": wp})
            p_raw.append(wp)
        p_holm = holm_correct(p_raw) if pending else []
        for rec, ph in zip(pending, p_holm):
            rec["p_holm"] = ph
            rows.append(rec)
    return pd.DataFrame(rows)


# ── Visualization ─────────────────────────────────────────────────────────

def save_overlay_grid(records: list, out_path: Path, n_samples: int = 4,
                      keys: "list | None" = None, title: str = "Frozen PanDerm attention — none vs color vs sam"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_image = {}
    for r in records:
        by_image.setdefault(r["image_path"], {})[r["backend"]] = r
    if keys is None:
        keys = list(by_image.keys())[:n_samples]
    else:
        keys = [k for k in keys if k in by_image]
    if not keys:
        print(f"  no matching images for {out_path.name} — skipping"); return

    # Leftmost column: the plain raw image, no heatmap — so it's clear where the
    # lesion actually is before judging any backend's attention overlay against it.
    ncols = 1 + len(BACKBONE_BACKENDS)
    fig, axes = plt.subplots(len(keys), ncols, figsize=(4 * ncols, 4 * len(keys)))
    axes = np.atleast_2d(axes)
    for row_i, img_path in enumerate(keys):
        any_rec = next(iter(by_image[img_path].values()))
        raw_pil = Image.open(img_path).convert("RGB")
        raw_resized = np.array(transforms.CenterCrop(224)(transforms.Resize(256)(raw_pil)))
        ax = axes[row_i, 0]
        ax.imshow(raw_resized)
        ax.set_title(f"raw image  pid={any_rec.get('patient_id', '?')}", fontsize=9)
        ax.axis("off")

        for col_i, backend in enumerate(BACKBONE_BACKENDS, start=1):
            ax = axes[row_i, col_i]
            rec = by_image[img_path].get(backend)
            if rec is None:
                ax.axis("off"); continue
            resized = rec["_resized"]
            attn = rec["_attn"]
            heat = (attn - attn.min()) / (attn.max() - attn.min() + 1e-9)
            heat_color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(resized, 0.6, heat_color, 0.4, 0)
            # keep the blacked-out background pure black — don't paint heatmap
            # color onto pixels the model never actually saw as content
            bg = resized.max(axis=-1) <= 8
            overlay[bg] = 0
            ax.imshow(overlay)
            ax.set_title(f"{backend}  pid={rec.get('patient_id', '?')}  ent={rec['entropy']:.2f}  "
                        f"top10={rec['top10_concentration']:.2f}", fontsize=9)
            ax.axis("off")
    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="images to sample")
    ap.add_argument("--backends", nargs="+", default=BACKBONE_BACKENDS)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--low-pigment-n", type=int, default=4,
                    help="also save a separate overlay grid for the N lowest-saturation "
                         "(color crop) images in the sample, for direct comparison")
    args = ap.parse_args()

    RES_DIR.mkdir(parents=True, exist_ok=True)
    images = sample_images(args.n, args.seed)
    print(f"Sampled {len(images)} images across {images['diagnosis'].nunique()} classes")
    print(images.groupby("diagnosis").size().to_dict())

    # Scored once here off the "color" crop (same convention as masking_experiment.py),
    # reused below both for the low-pigment-vs-high-pigment significance test and for
    # picking the low-pigment overlay grid's images.
    images["pigment_score"] = images.apply(pigment_score, axis=1)

    model = load_frozen_encoder()

    records = []
    for backend in args.backends:
        print(f"\n=== backend: {backend} ===")
        missing = 0
        for i, row in images.iterrows():
            p = cache_path(row, backend)
            if not p.exists():
                missing += 1
                continue
            native_grid, attn_map_224, resized, _ = raw_attention_map(model, p)
            m = attention_metrics(native_grid, attn_map_224, resized if backend != "none" else None)
            m.update(image_path=row["image_path"], diagnosis=row["diagnosis"], backend=backend,
                    patient_id=row.get("patient_id", row.get("folder_name", "?")))
            m["_attn"] = attn_map_224
            m["_resized"] = resized
            records.append(m)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(images)}")
        if missing:
            print(f"  {missing}/{len(images)} not cached for '{backend}'")

    if not records:
        print("No records produced — nothing to compare."); return

    df = pd.DataFrame(records)
    df = df.merge(images[["image_path", "pigment_score"]], on="image_path", how="left")
    plain_cols = [c for c in df.columns if not c.startswith("_")]
    df[plain_cols].to_csv(RES_DIR / "attention_metrics.csv", index=False)

    print("\n" + "=" * 80)
    print("ATTENTION CONCENTRATION — per backend (mean ± std)")
    print("=" * 80)
    metric_cols = ["entropy", "entropy_frac_of_max", "top10_concentration",
                  "centroid_offset", "fg_attention_fraction"]
    summary = df.groupby("backend")[metric_cols].agg(["mean", "std", "count"])
    print(summary)
    summary.to_csv(RES_DIR / "attention_summary.csv")

    print("\nFriedman (omnibus) + pairwise Wilcoxon (Holm-corrected):")
    all_posthoc = []
    for metric in ["entropy", "top10_concentration", "centroid_offset"]:
        omni, posthoc = friedman_and_posthoc(df, metric)
        print(f"\n  {metric}: Friedman chi2={omni['friedman_stat']:.3f}  "
             f"p={omni['friedman_p']:.4f}  (n={omni['n_images']} paired images)")
        for _, r in posthoc.iterrows():
            star = " *" if r["p_holm"] < 0.05 else ""
            print(f"    {r['a']:<8} vs {r['b']:<8}  delta={r['delta_a_minus_b']:+.4f}  "
                 f"p_holm={r['p_holm']:.4f}{star}")
        all_posthoc.append(posthoc)

    # fg_attention_fraction: color vs sam only (the metric "none" doesn't have)
    fg_df = df.dropna(subset=["fg_attention_fraction"])
    if fg_df["backend"].nunique() >= 2:
        wide = fg_df.pivot(index="image_path", columns="backend", values="fg_attention_fraction").dropna()
        if {"color", "sam"}.issubset(wide.columns):
            w, wp = stats.wilcoxon(wide["color"], wide["sam"])
            print(f"\n  fg_attention_fraction: color vs sam  "
                 f"delta={float((wide['color']-wide['sam']).mean()):+.4f}  p={wp:.4f}")

    pd.concat(all_posthoc, ignore_index=True).to_csv(RES_DIR / "pairwise_tests.csv", index=False)

    # Median-split low-vs-high pigment significance test, per backend — an actual
    # test of "low-pigment lesions get worse/more diffuse attention", instead of
    # eyeballing overlays. Independent-groups (Mann-Whitney), since low- and
    # high-pigment images are different lesions, not paired views of one image.
    print("\nLow-pigment vs. high-pigment attention (median split, Mann-Whitney, Holm-corrected):")
    pigment_tests = pigment_group_test(df.dropna(subset=["pigment_score"]),
                                       ["entropy", "top10_concentration", "fg_attention_fraction"])
    for _, r in pigment_tests.iterrows():
        star = " *" if r["p_holm"] < 0.05 else ""
        print(f"  {r['backend']:<8} {r['metric']:<22} low_median={r['low_median']:.4f}  "
             f"high_median={r['high_median']:.4f}  delta={r['delta_low_minus_high']:+.4f}  "
             f"p_holm={r['p_holm']:.4f}{star}")
    pigment_tests.to_csv(RES_DIR / "pigment_group_tests.csv", index=False)

    save_overlay_grid(records, RES_DIR / "attention_overlay_grid.png", n_samples=4)

    # Separate grid of just the lowest-pigment images in the sample, so they can be
    # compared across backends without scrolling through the general grid.
    low_pigment_paths = (images.dropna(subset=["pigment_score"])
                                .nsmallest(args.low_pigment_n, "pigment_score")["image_path"]
                                .tolist())
    save_overlay_grid(records, RES_DIR / "attention_overlay_grid_low_pigment.png",
                      keys=low_pigment_paths,
                      title="Frozen PanDerm attention — low-pigment lesions, none vs color vs sam")

    print(f"\nResults: {RES_DIR}")


if __name__ == "__main__":
    main()
