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

Metrics (comparable across all three conditions, no reference mask needed):
    entropy               Shannon entropy of the attention distribution over
                          patches, normalised to sum=1. Lower = more focused.
    top10_concentration   fraction of total attention mass sitting in the
                          10% most-attended patches. Higher = more focused.
    centroid_offset       distance of the attention-weighted centroid from
                          the frame centre, normalised by the frame diagonal.

Metric computed only for color/sam (self-referenced against their own saved
crop — non-black pixels ARE "foreground" by construction, no external
ground truth needed):
    fg_attention_fraction  fraction of attention mass landing on non-black
                          (kept) pixels within that condition's own crop.
                          Answers: even within what the model was shown
                          post-crop, is attention still landing on real
                          content, or partly wasted on the blacked-out
                          background padding?

Statistics: paired across the 3 conditions on identical images — Friedman
test (omnibus) + pairwise Wilcoxon signed-rank with Holm correction, same
framework as the segmentation-quality discussion.

Masking-robustness experiment (--masking-experiment): tests whether attention
is more fragile on low-pigment lesions specifically, per Oscar's feedback.
Masks 20% of the 224x224 image in 16x16 patches (filled with the image's own
mean color, NOT black — avoids recreating the background-bias artifact found
earlier) on a stratified sample of well-pigmented vs. low-pigment lesions
(picked by mean HSV saturation within the crop), then compares masked vs.
unmasked attention within each group and the *degradation* between groups.

Usage:
    python panderm/attention_comparison.py --n 100
    python panderm/attention_comparison.py --n 100 --backends none color sam
    python panderm/attention_comparison.py --masking-experiment --mask-n-per-group 30
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
    resized_pil = pil_img.resize((224, 224))
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
        # Holm-Bonferroni across this metric's pairwise family
        order = posthoc["p_raw"].argsort()
        m = len(posthoc)
        holm = np.empty(m)
        running_max = 0.0
        for rank, idx in enumerate(order):
            adj = (m - rank) * posthoc["p_raw"].iloc[idx]
            running_max = max(running_max, adj)
            holm[idx] = min(running_max, 1.0)
        posthoc["p_holm"] = holm
    return omnibus, posthoc


# ── Visualization ─────────────────────────────────────────────────────────

def save_overlay_grid(records: list, out_path: Path, n_samples: int = 4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_image = {}
    for r in records:
        by_image.setdefault(r["image_path"], {})[r["backend"]] = r
    keys = list(by_image.keys())[:n_samples]

    fig, axes = plt.subplots(len(keys), len(BACKBONE_BACKENDS),
                             figsize=(4 * len(BACKBONE_BACKENDS), 4 * len(keys)))
    axes = np.atleast_2d(axes)
    for row_i, img_path in enumerate(keys):
        for col_i, backend in enumerate(BACKBONE_BACKENDS):
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
            ax.set_title(f"{backend}  ent={rec['entropy']:.2f}  "
                        f"top10={rec['top10_concentration']:.2f}", fontsize=9)
            ax.axis("off")
    plt.suptitle("Frozen PanDerm attention — none vs color vs sam", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Masking-robustness experiment (well-pigmented vs. low-pigment) ───────

def pigment_score(row: pd.Series, backend: str = "color") -> float:
    """Mean HSV saturation within the crop's foreground (backend != 'none')
    or the whole resized frame (backend == 'none'). Low saturation ~ a
    low-pigment lesion. Objective stand-in for eyeballing "less pigment"."""
    p = cache_path(row, backend)
    if not p.exists():
        return np.nan
    img = np.array(Image.open(p).convert("RGB").resize((224, 224)))
    fg = img.max(axis=-1) > 8 if backend != "none" else np.ones(img.shape[:2], dtype=bool)
    if not fg.any():
        return np.nan
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return float(hsv[..., 1][fg].mean())


def build_pigment_groups(n_per_group: int, seed: int, backend: str = "color"
                         ) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Stratifies the full manifest by pigment_score and returns the
    n_per_group lowest- and highest-saturation images as (low, high)."""
    manifest = pd.read_csv(OUTPUT_DIR / "dataset_manifest.csv")
    manifest["pigment_score"] = manifest.apply(lambda r: pigment_score(r, backend), axis=1)
    manifest = manifest.dropna(subset=["pigment_score"])
    low = manifest.nsmallest(n_per_group, "pigment_score").reset_index(drop=True)
    high = manifest.nlargest(n_per_group, "pigment_score").reset_index(drop=True)
    return low, high


def run_masking_experiment(model: torch.nn.Module, n_per_group: int, mask_frac: float,
                           patch_size: int, backend: str, seed: int) -> list:
    """Runs each sampled image both unmasked and masked; returns one record
    per (image, masked) pair, tagged with its pigment_group."""
    low, high = build_pigment_groups(n_per_group, seed, backend=backend)
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
                     masked=masked)
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
            ax.set_title(f"{entry['group']}-pigment  {tag}  ent={rec['entropy']:.2f}", fontsize=9)
            ax.axis("off")
    plt.suptitle("20%% patch masking — attention, low- vs high-pigment lesions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def run_masking_main(model: torch.nn.Module, args) -> None:
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

    metric_cols = ["entropy", "top10_concentration", "fg_attention_fraction", "attention_on_masked"]
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


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="images to sample")
    ap.add_argument("--backends", nargs="+", default=BACKBONE_BACKENDS)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--masking-experiment", action="store_true",
                    help="run the 20%% patch-masking robustness test (well- vs. low-pigment) instead of the backend comparison")
    ap.add_argument("--mask-frac", type=float, default=0.2)
    ap.add_argument("--patch-size", type=int, default=16)
    ap.add_argument("--mask-n-per-group", type=int, default=30)
    ap.add_argument("--mask-backend", default="color", help="which cached crop to run the masking test on")
    args = ap.parse_args()

    if args.masking_experiment:
        model = load_frozen_encoder()
        run_masking_main(model, args)
        return

    RES_DIR.mkdir(parents=True, exist_ok=True)
    images = sample_images(args.n, args.seed)
    print(f"Sampled {len(images)} images across {images['diagnosis'].nunique()} classes")
    print(images.groupby("diagnosis").size().to_dict())

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
            m.update(image_path=row["image_path"], diagnosis=row["diagnosis"], backend=backend)
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
    save_overlay_grid(records, RES_DIR / "attention_overlay_grid.png", n_samples=4)

    print(f"\nResults: {RES_DIR}")


if __name__ == "__main__":
    main()
