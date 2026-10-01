"""
PanDerm Feature Extraction — frozen backbone
=============================================
Runs the pretrained PanDerm checkpoint (unchanged weights) over every image,
extracts CLS token features, aggregates to patient level, and saves to disk
for evaluate.py --mode frozen.

Frozen only. Fine-tuned models are evaluated with their own head on each
fold's test patients (panderm_finetuning.py writes test_predictions_fold{i}.csv),
not through features here.

Requires GPU. Run this before evaluate.py --mode frozen.

Usage:
    python extract_features.py

Outputs (saved to FEATURES_DIR):
    fold{i}_image_features.npy      — per-image CLS features  (N_images, 1024)
    patient_features_fold{i}.npy    — patient-level features   (N_patients, 1024)
    patient_labels.npy              — patient labels
    patient_group_ids.npy           — patient group IDs
    group_fold_mapping.csv          — fold assignment per patient group
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from collections import OrderedDict

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PANDERM_CLASS, CHECKPOINT_LARGE,
    OUTPUT_DIR, SEGMENTATION_BACKEND,
    CLASS_LABELS, IMAGENET_MEAN, IMAGENET_STD,
    NB_CLASSES, BATCH_SIZE, NUM_WORKERS, N_FOLDS, input_image,
)

# --mode is kept so existing commands keep working; only "frozen" is valid now.
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["frozen"], default="frozen")
args, _ = parser.parse_known_args()

# Mode first, then backend — results/<mode>/<backend>/, features/<mode>/<backend>/.
FEATURES_DIR = OUTPUT_DIR / "features" / "frozen" / SEGMENTATION_BACKEND
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ── Model ─────────────────────────────────────────────────────────────────

def load_panderm_encoder(ckpt_path: Path) -> torch.nn.Module:
    for p in [str(PANDERM_CLASS.parent), str(PANDERM_CLASS)]:
        if p not in sys.path: sys.path.insert(0, p)
    from models.modeling_finetune import panderm_large_patch16_224
    model = panderm_large_patch16_224(num_classes=NB_CLASSES)
    ckpt  = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    clean = OrderedDict(
        (k.replace("encoder.", "").replace("module.", ""), v)
        for k, v in state.items()
    )
    missing, unexpected = model.load_state_dict(clean, strict=False)
    print(f"    missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()
    return model.to(DEVICE)


# ── Dataset ───────────────────────────────────────────────────────────────

class DermDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        try:    img = Image.open(self.paths[idx]).convert("RGB")
        except: img = Image.new("RGB", (224, 224))
        return self.transform(img), idx


eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


@torch.no_grad()
def extract_features(model: torch.nn.Module, paths: list) -> np.ndarray:
    ds     = DermDataset(paths, eval_tf)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS)
    feats  = np.zeros((len(paths), 1024), dtype=np.float32)
    for imgs, idxs in loader:
        f = model.forward_features(imgs.to(DEVICE), is_train=False)
        if isinstance(f, tuple): f = f[0]
        if f.dim() > 2: f = f.mean(dim=1)
        feats[idxs.numpy()] = f.cpu().numpy()
    return feats


# ── Manifest ──────────────────────────────────────────────────────────────

def load_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(OUTPUT_DIR / "dataset_manifest.csv")

    # Raw image for "none", segmented_cache/<backend>/ crop otherwise — chosen
    # from config, not the manifest's segmented_path (last backend segmented).
    manifest["input_image"] = manifest["image_path"].map(input_image)
    missing = ~manifest["input_image"].apply(lambda p: Path(p).exists())
    if missing.any():
        raise FileNotFoundError(
            f"{missing.sum()} images for backend '{SEGMENTATION_BACKEND}' not found, "
            f"e.g. {manifest.loc[missing, 'input_image'].iloc[0]}")
    print(f"  Backend '{SEGMENTATION_BACKEND}': e.g. {manifest['input_image'].iloc[0]}")

    if "label" not in manifest.columns:
        manifest["label"] = manifest["diagnosis"].map(CLASS_LABELS)

    print(f"Manifest: {len(manifest)} images, {manifest['patient_id'].nunique()} patients")
    return manifest


# ── Aggregation ───────────────────────────────────────────────────────────

def aggregate_to_patient(features: np.ndarray, df: pd.DataFrame):
    df = df.copy().reset_index(drop=True)
    df["patient_id"] = df["patient_id"].astype(str)
    df["group_id"]   = df["patient_id"] + "__" + df["diagnosis"]
    groups   = sorted(df["group_id"].unique())
    p_feats  = np.zeros((len(groups), features.shape[1]), dtype=np.float32)
    p_labels = np.zeros(len(groups), dtype=np.int64)
    for i, gid in enumerate(groups):
        mask = (df["group_id"] == gid).values
        p_feats[i]  = features[mask].mean(axis=0)
        p_labels[i] = df.loc[mask, "label"].iloc[0]
    print(f"  {len(df)} images -> {len(groups)} patient groups")
    return p_feats, p_labels, groups


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    manifest = load_manifest()

    # ── Image-level feature extraction ────────────────────────────────────
    # Extract once with pretrained checkpoint, reuse across all folds
    print(f"\nFrozen mode: extracting with pretrained checkpoint")
    print(f"  {CHECKPOINT_LARGE}")
    shared_cache = FEATURES_DIR / "frozen_image_features.npy"
    if shared_cache.exists():
        print("  Cached — skipping extraction")
        feats = np.load(str(shared_cache))
    else:
        model = load_panderm_encoder(CHECKPOINT_LARGE)
        feats = extract_features(model, manifest["input_image"].tolist())
        np.save(str(shared_cache), feats)
        del model; torch.cuda.empty_cache()
        print(f"  Saved {shared_cache.name}  {feats.shape}")

    for fold_idx in range(N_FOLDS):
        cache = FEATURES_DIR / f"fold{fold_idx}_image_features.npy"
        if not cache.exists():
            np.save(str(cache), feats)
            print(f"  Fold {fold_idx}: linked frozen features")

    # ── Aggregate to patient level ────────────────────────────────────────
    manifest["patient_id"] = manifest["patient_id"].astype(str)
    group_fold = manifest.groupby(
        manifest["patient_id"] + "__" + manifest["diagnosis"]
    )["fold"].first()
    fold_assignments = dict(zip(group_fold.index, group_fold.values))

    ref_labels = ref_groups = None
    for fold_idx in range(N_FOLDS):
        cache = FEATURES_DIR / f"fold{fold_idx}_image_features.npy"
        if not cache.exists(): continue
        p_feats, p_labels, group_ids = aggregate_to_patient(
            np.load(str(cache)), manifest
        )
        np.save(str(FEATURES_DIR / f"patient_features_fold{fold_idx}.npy"), p_feats)
        ref_labels, ref_groups = p_labels, group_ids

    np.save(str(FEATURES_DIR / "patient_labels.npy"),    ref_labels)
    np.save(str(FEATURES_DIR / "patient_group_ids.npy"), np.array(ref_groups))
    pd.DataFrame({
        "group_id": list(fold_assignments.keys()),
        "fold":     list(fold_assignments.values()),
    }).to_csv(FEATURES_DIR / "group_fold_mapping.csv", index=False)

    print(f"\nAll features saved to: {FEATURES_DIR}")


if __name__ == "__main__":
    main()
