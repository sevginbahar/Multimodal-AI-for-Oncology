"""
PanDerm Feature Extraction — mask-guided patch pooling (frozen)
================================================================
Segmentation as GUIDANCE instead of a crop. The FULL RAW image goes through
frozen PanDerm exactly as in extract_features.py (no crop, no black
background, all context available to self-attention). Only the image
feature changes:

    PanDerm-L outputs 197 tokens x 1024 after the final norm: token 0 = CLS,
    tokens 1..196 = one per 16x16 patch of the 224x224 input (14x14 grid).
    extract_features.py keeps token 0. Here the image feature is the average
    of the 196 patch tokens, each weighted by the fraction of its patch that
    the lesion mask covers — skin patches get weight 0.

The mask comes from config.SEGMENTATION_BACKEND's segmenter (e.g. "sam_v2")
run on the raw image, gets the same Resize(256) + CenterCrop(224) as the
image so the 14x14 weight grid lines up with the patch grid, and is cached
as a PNG so re-runs skip segmentation. If no lesion pixel falls inside the
centre crop, all patches get weight 1 (plain patch mean) and the image is
counted in mask_weight_stats.csv.

Outputs (same files as extract_features.py writes for frozen mode):
    features/maskguided/<backend>/
        masks/...                    cached masks
        fold{i}_image_features.npy, patient_features_fold{i}.npy,
        patient_labels.npy, patient_group_ids.npy, group_fold_mapping.csv
        mask_weight_stats.csv

Control (--pooling mean): every patch gets weight 1 — the plain average of the
196 patch tokens, no mask. Separates "CLS -> patch pooling" from "lesion
weighting". Requires SEGMENTATION_BACKEND = "none" (no segmentation involved)
and writes to features/meanall/none/.

Usage:
    python panderm/extract_features_maskguided.py                  # mask-guided
    python panderm/evaluate.py --mode maskguided
    python panderm/extract_features_maskguided.py --pooling mean   # control, backend "none"
    python panderm/evaluate.py --mode meanall
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from collections import OrderedDict

from config import (OUTPUT_DIR, DATA_ROOT, SEGMENTATION_BACKEND, N_FOLDS,
                    IMAGENET_MEAN, IMAGENET_STD, CLASS_LABELS,
                    PANDERM_CLASS, CHECKPOINT_LARGE, NB_CLASSES)

GRID, PATCH = 14, 16
MASK_DIR = OUTPUT_DIR / "features" / "maskguided" / SEGMENTATION_BACKEND / "masks"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_resize_crop = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
_resize_crop_mask = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.CenterCrop(224)])
_to_tensor = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def load_frozen_encoder() -> torch.nn.Module:
    """Pretrained PanDerm-L, same loading as extract_features.py (frozen, eval mode)."""
    for p in [str(PANDERM_CLASS.parent), str(PANDERM_CLASS)]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from models.modeling_finetune import panderm_large_patch16_224
    model = panderm_large_patch16_224(num_classes=NB_CLASSES)
    ckpt = torch.load(str(CHECKPOINT_LARGE), map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    clean = OrderedDict((k.replace("encoder.", "").replace("module.", ""), v) for k, v in state.items())
    missing, unexpected = model.load_state_dict(clean, strict=False)
    print(f"Frozen encoder loaded: missing={len(missing)} unexpected={len(unexpected)} "
          f"(missing = new head, unexpected = SSL decoder — both expected)")
    return model.eval().to(DEVICE)


@torch.no_grad()
def all_tokens(model, x: torch.Tensor) -> torch.Tensor:
    """PanDerm's forward_features() up to and including the final norm, but
    returning all 197 tokens instead of only token 0 (CLS)."""
    x = model.patch_embed(x)
    b = x.size(0)
    x = torch.cat((model.cls_token.expand(b, -1, -1), x), dim=1)
    x = x + model.pos_embed.expand(b, -1, -1).type_as(x).to(x.device)
    x = model.pos_drop(x)
    rel = model.rel_pos_bias() if model.rel_pos_bias is not None else None
    for blk in model.blocks:
        x = blk(x, rel_pos_bias=rel)
    return model.norm(x)


def patch_weights(mask_224: np.ndarray) -> np.ndarray:
    """224x224 binary mask -> 196 weights = fraction of lesion pixels per 16x16 patch,
    in the same row-major order as PanDerm's patch tokens."""
    return mask_224.reshape(GRID, PATCH, GRID, PATCH).mean(axis=(1, 3)).flatten()


def get_mask(row: pd.Series, image: np.ndarray, segmenter_factory) -> np.ndarray:
    p = (MASK_DIR / Path(row["image_path"]).relative_to(DATA_ROOT)).with_suffix(".png")
    if p.exists():
        return np.array(Image.open(p)) > 0
    mask = segmenter_factory().segment(image) > 0
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(p)
    return mask


def aggregate_to_patient(features: np.ndarray, df: pd.DataFrame):
    """Same patient grouping / ordering as extract_features.aggregate_to_patient()."""
    gids = df["patient_id"].astype(str) + "__" + df["diagnosis"]
    groups = sorted(gids.unique())
    p_feats = np.stack([features[(gids == g).values].mean(axis=0) for g in groups]).astype(np.float32)
    p_labels = np.array([df.loc[(gids == g).values, "label"].iloc[0] for g in groups], dtype=np.int64)
    return p_feats, p_labels, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooling", choices=["mask", "mean"], default="mask",
                    help="mask = lesion-weighted patch average; mean = all patches weight 1 (control)")
    use_mask = ap.parse_args().pooling == "mask"

    if use_mask and SEGMENTATION_BACKEND == "none":
        raise SystemExit('Mask-guided pooling needs masks: set SEGMENTATION_BACKEND in config.py '
                         'to a segmenter (e.g. "sam_v2").')
    if not use_mask and SEGMENTATION_BACKEND != "none":
        raise SystemExit('--pooling mean uses no mask: set SEGMENTATION_BACKEND = "none" in config.py.')
    FEATURES_DIR = OUTPUT_DIR / "features" / ("maskguided" if use_mask else "meanall") / SEGMENTATION_BACKEND
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(OUTPUT_DIR / "dataset_manifest.csv")
    if "label" not in manifest.columns:
        manifest["label"] = manifest["diagnosis"].map(CLASS_LABELS)
    print(f"Manifest: {len(manifest)} images | raw images through PanDerm | pooling: "
          f"{'lesion-weighted (' + SEGMENTATION_BACKEND + ' masks)' if use_mask else 'mean of all 196 patches'}"
          f" -> {FEATURES_DIR}")

    _seg = {}
    def segmenter():   # built lazily: not needed at all once every mask is cached
        if "s" not in _seg:
            from segment_lesions import build_segmenter
            _seg["s"] = build_segmenter(SEGMENTATION_BACKEND)
        return _seg["s"]

    model = load_frozen_encoder()
    feats = np.zeros((len(manifest), 1024), dtype=np.float32)
    stats = []
    for i, row in manifest.iterrows():
        pil = Image.open(row["image_path"]).convert("RGB")
        mask = get_mask(row, np.array(pil), segmenter) if use_mask else np.ones((pil.height, pil.width), bool)

        x = _to_tensor(_resize_crop(pil)).unsqueeze(0).to(DEVICE)
        tokens = all_tokens(model, x)[0].float().cpu().numpy()          # 197 x 1024
        if i == 0:   # sanity check: token 0 must equal what extract_features.py uses
            with torch.no_grad():
                ref = model.forward_features(x, is_train=False)[0].float().cpu().numpy()
            assert np.allclose(tokens[0], ref, atol=1e-4), "CLS token mismatch vs forward_features"
        patches = tokens[1:]                                              # 196 x 1024

        m224 = np.array(_resize_crop_mask(Image.fromarray(mask.astype(np.uint8) * 255))) > 127
        w = patch_weights(m224)
        empty = w.sum() <= 0
        if empty:
            w = np.ones_like(w)
        feats[i] = (w[:, None] * patches).sum(axis=0) / w.sum()

        stats.append({"image_path": row["image_path"], "diagnosis": row["diagnosis"],
                      "lesion_frac_full": float(mask.mean()), "lesion_frac_crop": float(m224.mean()),
                      "n_lesion_patches": 0 if empty else int((w > 0).sum()),
                      "empty_mask_fallback": empty})
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(manifest)}]")

    st = pd.DataFrame(stats)
    st.to_csv(FEATURES_DIR / "mask_weight_stats.csv", index=False)
    print(f"\nEmpty-mask fallbacks (no lesion in the centre crop): {int(st.empty_mask_fallback.sum())}")
    print("Median lesion patches per image by class:",
          st.groupby("diagnosis").n_lesion_patches.median().to_dict())

    # Same files as extract_features.py writes for frozen mode
    np.save(FEATURES_DIR / "frozen_image_features.npy", feats)
    p_feats, p_labels, groups = aggregate_to_patient(feats, manifest)
    for f in range(N_FOLDS):   # frozen backbone: identical features for every fold
        np.save(FEATURES_DIR / f"fold{f}_image_features.npy", feats)
        np.save(FEATURES_DIR / f"patient_features_fold{f}.npy", p_feats)
    np.save(FEATURES_DIR / "patient_labels.npy", p_labels)
    np.save(FEATURES_DIR / "patient_group_ids.npy", np.array(groups))
    gid = manifest["patient_id"].astype(str) + "__" + manifest["diagnosis"]
    fold_map = manifest.groupby(gid)["fold"].first()
    pd.DataFrame({"group_id": fold_map.index, "fold": fold_map.values}).to_csv(
        FEATURES_DIR / "group_fold_mapping.csv", index=False)
    print(f"{len(manifest)} images -> {len(groups)} patient groups. Features saved to: {FEATURES_DIR}")
    print(f"Next: python panderm/evaluate.py --mode {'maskguided' if use_mask else 'meanall'}")


if __name__ == "__main__":
    main()
