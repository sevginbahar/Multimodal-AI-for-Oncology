"""
PanDerm Feature Extraction — 5-Fold Cross-Validation
=====================================================
Loads fold-specific fine-tuned PanDerm checkpoints, extracts CLS token
features for every image, aggregates to patient level, and saves to disk.

Requires GPU. Run this before evaluate.py.

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
    DATA_ROOT, SEGMENTED_DIR, OUTPUT_DIR, FEATURES_DIR,
    CLASS_LABELS, IMAGENET_MEAN, IMAGENET_STD,
    NB_CLASSES, BATCH_SIZE, NUM_WORKERS, N_FOLDS,
)

FEATURES_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ── Model ─────────────────────────────────────────────────────────────────

def find_best_checkpoint(fold_dir: Path) -> Path:
    for name in ["checkpoint-best.pth", "best.pth"]:
        p = fold_dir / name
        if p.exists(): return p
    ckpts = sorted(fold_dir.glob("checkpoint-*.pth"),
                   key=lambda p: int(p.stem.split("-")[-1]))
    if ckpts: return ckpts[-1]
    raise FileNotFoundError(f"No checkpoint found in {fold_dir}")


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

    cluster_prefix     = manifest["image_path"].iloc[0].rsplit("/dermoscopy", 1)[0]
    cluster_seg_prefix = manifest["segmented_path"].iloc[0].rsplit("/segmented_cache", 1)[0]

    manifest["image"] = manifest["image_path"].str.replace(
        cluster_prefix + "/dermoscopy", str(DATA_ROOT), regex=False)
    manifest["segmented_image"] = manifest["segmented_path"].str.replace(
        cluster_seg_prefix + "/segmented_cache", str(SEGMENTED_DIR), regex=False)

    manifest["input_image"] = manifest["segmented_image"]
    missing = ~manifest["input_image"].apply(lambda p: Path(p).exists())
    if missing.sum() > 0:
        print(f"  {missing.sum()} segmented images missing — falling back to originals")
        manifest.loc[missing, "input_image"] = manifest.loc[missing, "image"]

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

    # ── Per-fold image-level feature extraction ───────────────────────────
    for fold_idx in range(N_FOLDS):
        cache = FEATURES_DIR / f"fold{fold_idx}_image_features.npy"
        if cache.exists():
            print(f"Fold {fold_idx}: cached — skipping extraction")
            continue

        fold_dir = OUTPUT_DIR / f"results_fold{fold_idx}"
        try:
            ckpt_path = find_best_checkpoint(fold_dir)
        except FileNotFoundError as e:
            print(f"Fold {fold_idx}: {e} — skipping"); continue

        print(f"\nFold {fold_idx}: loading {ckpt_path.name}")
        model = load_panderm_encoder(ckpt_path)
        feats = extract_features(model, manifest["input_image"].tolist())
        np.save(str(cache), feats)
        del model; torch.cuda.empty_cache()
        print(f"  Saved {cache.name}  {feats.shape}")

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
