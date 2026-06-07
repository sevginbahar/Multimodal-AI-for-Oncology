"""
PanDerm Feature Extraction + K-Fold Evaluation
===============================================
Loads fold-specific fine-tuned PanDerm checkpoints, extracts CLS token features,
aggregates to patient level, and evaluates with logistic regression.

Usage:
    python extract_and_evaluate.py

Outputs (saved to CONFIG["output_dir"]/features/):
    fold{i}_image_features.npy      — per-image features (562, 1024)
    patient_features_fold{i}.npy    — patient-level features (166, 1024)
    patient_labels.npy              — patient labels
    patient_group_ids.npy           — patient group IDs
    group_fold_mapping.csv          — fold assignment per patient group
    kfold_results_finetuned.csv     — aggregate results
    fold_{i}_report.txt             — per-fold classification reports
"""

import sys
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score,
    classification_report, confusion_matrix,
    roc_auc_score, cohen_kappa_score,
)
from sklearn.preprocessing import label_binarize

# ── Config ────────────────────────────────────────────────────────────────
CONFIG = {
    "panderm_repo":    "/path/to/PanDerm/classification",
    "checkpoint":      "/path/to/panderm_ll_data6_checkpoint-499.pth",
    "manifest":        "/path/to/dataset_manifest.csv",
    "image_root":      "/path/to/dermoscopy",
    "segmented_cache": "/path/to/segmented_cache",
    "output_dir":      "./results",
    "nb_classes":      3,
    "batch_size":      32,
    "num_workers":     4,
    "n_folds":         5,
}

CLASS_NAMES   = ["DN", "MIA", "Minsitu"]
DISPLAY_NAMES = {"DN": "Dysplastic Nevus", "MIA": "Melanoma Stage IA", "Minsitu": "Melanoma In Situ"}
CLASS_LABELS  = {"DN": 0, "MIA": 1, "Minsitu": 2}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.228, 0.224, 0.225]   # PanDerm uses 0.228 not 0.229
FEATURES_DIR  = Path(CONFIG["output_dir"]) / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ── Helpers ───────────────────────────────────────────────────────────────
def find_best_checkpoint(fold_dir: Path) -> Path:
    for name in ["checkpoint-best.pth", "best.pth"]:
        p = fold_dir / name
        if p.exists(): return p
    ckpts = sorted(fold_dir.glob("checkpoint-*.pth"),
                   key=lambda p: int(p.stem.split("-")[-1]))
    if ckpts: return ckpts[-1]
    raise FileNotFoundError(f"No checkpoint found in {fold_dir}")


def load_panderm_encoder(ckpt_path: Path) -> torch.nn.Module:
    for p in [str(Path(CONFIG["panderm_repo"]).parent), CONFIG["panderm_repo"]]:
        if p not in sys.path: sys.path.insert(0, p)
    from models.modeling_finetune import panderm_large_patch16_224
    model = panderm_large_patch16_224(num_classes=CONFIG["nb_classes"])
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
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


@torch.no_grad()
def extract_features(model: torch.nn.Module, paths: list, batch_size: int = 32) -> np.ndarray:
    ds     = DermDataset(paths, eval_tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=CONFIG["num_workers"])
    feats  = np.zeros((len(paths), 1024), dtype=np.float32)
    for imgs, idxs in loader:
        f = model.forward_features(imgs.to(DEVICE), is_train=False)
        if isinstance(f, tuple): f = f[0]
        if f.dim() > 2: f = f.mean(dim=1)
        feats[idxs.numpy()] = f.cpu().numpy()
    return feats


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


# ── Load manifest ─────────────────────────────────────────────────────────
def load_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(CONFIG["manifest"])

    cluster_prefix     = manifest["image_path"].iloc[0].rsplit("/dermoscopy", 1)[0]
    cluster_seg_prefix = manifest["segmented_path"].iloc[0].rsplit("/segmented_cache", 1)[0]

    manifest["image"] = manifest["image_path"].str.replace(
        cluster_prefix + "/dermoscopy", CONFIG["image_root"], regex=False)
    manifest["segmented_image"] = manifest["segmented_path"].str.replace(
        cluster_seg_prefix + "/segmented_cache", CONFIG["segmented_cache"], regex=False)

    manifest["input_image"] = manifest["segmented_image"]
    missing = ~manifest["input_image"].apply(lambda p: Path(p).exists())
    if missing.sum() > 0:
        print(f"  {missing.sum()} segmented images missing — falling back to originals")
        manifest.loc[missing, "input_image"] = manifest.loc[missing, "image"]

    if "label" not in manifest.columns:
        manifest["label"] = manifest["diagnosis"].map(CLASS_LABELS)

    print(f"Manifest: {len(manifest)} images, {manifest['patient_id'].nunique()} patients")
    return manifest


# ── K-fold evaluation ─────────────────────────────────────────────────────
def run_kfold_evaluation(patient_labels, group_ids, fold_assignments):
    fold_results = []

    for fold_idx in range(CONFIG["n_folds"]):
        feat_path = FEATURES_DIR / f"patient_features_fold{fold_idx}.npy"
        if not feat_path.exists():
            print(f"  Fold {fold_idx}: features missing — skipping")
            continue

        patient_features = np.load(str(feat_path))
        train_mask = np.array([fold_assignments[gid] != fold_idx for gid in group_ids])
        test_mask  = ~train_mask
        X_tr, y_tr = patient_features[train_mask], patient_labels[train_mask]
        X_te, y_te = patient_features[test_mask],  patient_labels[test_mask]

        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                                 solver="lbfgs", random_state=42, multi_class="multinomial")
        clf.fit(X_tr, y_tr)
        y_pred  = clf.predict(X_te)
        y_proba = clf.predict_proba(X_te)

        y_bin  = label_binarize(y_te, classes=[0, 1, 2])
        report = classification_report(y_te, y_pred, target_names=CLASS_NAMES,
                                       output_dict=True, zero_division=0)
        try:    macro_auc = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
        except: macro_auc = float("nan")

        per_class_auc = {}
        for ci, cls in enumerate(CLASS_NAMES):
            try:    per_class_auc[cls] = roc_auc_score(y_bin[:, ci], y_proba[:, ci])
            except: per_class_auc[cls] = float("nan")

        result = {
            "fold":                fold_idx,
            "balanced_accuracy":   balanced_accuracy_score(y_te, y_pred),
            "accuracy":            accuracy_score(y_te, y_pred),
            "macro_auc":           macro_auc,
            "cohen_kappa":         cohen_kappa_score(y_te, y_pred),
            "macro_f1":            report["macro avg"]["f1-score"],
            "per_class_auc":       per_class_auc,
            "per_class_f1":        {c: report[c]["f1-score"]  for c in CLASS_NAMES},
            "per_class_precision": {c: report[c]["precision"] for c in CLASS_NAMES},
            "per_class_recall":    {c: report[c]["recall"]    for c in CLASS_NAMES},
            "confusion_matrix":    confusion_matrix(y_te, y_pred, labels=[0, 1, 2]),
            "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
            "report_text": classification_report(y_te, y_pred,
                                                 target_names=CLASS_NAMES, zero_division=0),
        }
        fold_results.append(result)
        print(f"  Fold {fold_idx+1}: BalAcc={result['balanced_accuracy']:.3f}  "
              f"AUC={macro_auc:.3f}  F1={result['macro_f1']:.3f}  "
              f"(train={result['n_train']}, test={result['n_test']})")

    return fold_results


def print_summary(fold_results):
    scalar_metrics = ["balanced_accuracy", "accuracy", "macro_auc", "cohen_kappa", "macro_f1"]
    agg = {}
    for m in scalar_metrics:
        vals = [r[m] for r in fold_results]
        agg[f"{m}_mean"] = np.nanmean(vals)
        agg[f"{m}_std"]  = np.nanstd(vals)
    for cls in CLASS_NAMES:
        for mt in ["f1", "precision", "recall", "auc"]:
            vals = [r[f"per_class_{mt}"][cls] for r in fold_results]
            agg[f"{cls}_{mt}_mean"] = np.nanmean(vals)
            agg[f"{cls}_{mt}_std"]  = np.nanstd(vals)

    print("\n" + "="*70)
    print("RESULTS -- Fine-Tuned PanDerm (Patient-Level, 5-Fold CV)")
    print("="*70)
    print(f"  Balanced Accuracy : {agg['balanced_accuracy_mean']:.3f} +/- {agg['balanced_accuracy_std']:.3f}")
    print(f"  Accuracy          : {agg['accuracy_mean']:.3f} +/- {agg['accuracy_std']:.3f}")
    print(f"  Macro AUC         : {agg['macro_auc_mean']:.3f} +/- {agg['macro_auc_std']:.3f}")
    print(f"  Macro F1          : {agg['macro_f1_mean']:.3f} +/- {agg['macro_f1_std']:.3f}")
    print(f"  Cohen Kappa       : {agg['cohen_kappa_mean']:.3f} +/- {agg['cohen_kappa_std']:.3f}")
    print(f"\n  {'Class':<22} {'F1':>14} {'AUC':>14}")
    print("  " + "-"*50)
    for cls in CLASS_NAMES:
        f1 = f"{agg[f'{cls}_f1_mean']:.3f}+/-{agg[f'{cls}_f1_std']:.3f}"
        au = f"{agg[f'{cls}_auc_mean']:.3f}+/-{agg[f'{cls}_auc_std']:.3f}"
        print(f"  {DISPLAY_NAMES[cls]:<22} {f1:>14} {au:>14}")
    print("="*70)
    return agg


def save_results(fold_results, agg):
    out = Path(CONFIG["output_dir"])
    rows = []
    for r in fold_results:
        row = {k: r[k] for k in ["fold","balanced_accuracy","accuracy",
                                   "macro_auc","macro_f1","cohen_kappa","n_train","n_test"]}
        for cls in CLASS_NAMES:
            row[f"{cls}_f1"]  = r["per_class_f1"][cls]
            row[f"{cls}_auc"] = r["per_class_auc"][cls]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "kfold_results_finetuned.csv", index=False)
    for r in fold_results:
        (out / f"fold_{r['fold']}_report.txt").write_text(
            f"Fold {r['fold']} -- train={r['n_train']} test={r['n_test']}\n\n" + r["report_text"])
    print(f"\nSaved results to {out}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    manifest = load_manifest()

    # Extract fold-specific features
    for fold_idx in range(CONFIG["n_folds"]):
        cache = FEATURES_DIR / f"fold{fold_idx}_image_features.npy"
        if cache.exists():
            print(f"Fold {fold_idx}: cached, skipping extraction")
            continue
        fold_dir = Path(CONFIG["output_dir"]) / f"results_fold{fold_idx}"
        try:
            ckpt_path = find_best_checkpoint(fold_dir)
        except FileNotFoundError as e:
            print(f"Fold {fold_idx}: {e} -- skipping"); continue
        print(f"\nFold {fold_idx}: {ckpt_path.name}")
        model = load_panderm_encoder(ckpt_path)
        feats = extract_features(model, manifest["input_image"].tolist())
        np.save(str(cache), feats)
        del model; torch.cuda.empty_cache()
        print(f"  Saved {cache.name}  {feats.shape}")

    # Aggregate to patient level
    manifest["patient_id"] = manifest["patient_id"].astype(str)
    group_fold = manifest.groupby(manifest["patient_id"] + "__" + manifest["diagnosis"])["fold"].first()
    fold_assignments = dict(zip(group_fold.index, group_fold.values))

    ref_labels = ref_groups = None
    for fold_idx in range(CONFIG["n_folds"]):
        cache = FEATURES_DIR / f"fold{fold_idx}_image_features.npy"
        if not cache.exists(): continue
        p_feats, p_labels, group_ids = aggregate_to_patient(np.load(str(cache)), manifest)
        np.save(str(FEATURES_DIR / f"patient_features_fold{fold_idx}.npy"), p_feats)
        ref_labels, ref_groups = p_labels, group_ids

    np.save(str(FEATURES_DIR / "patient_labels.npy"),    ref_labels)
    np.save(str(FEATURES_DIR / "patient_group_ids.npy"), np.array(ref_groups))
    fold_df = pd.DataFrame({"group_id": list(fold_assignments.keys()),
                             "fold":     list(fold_assignments.values())})
    fold_df.to_csv(FEATURES_DIR / "group_fold_mapping.csv", index=False)
    print("Patient-level features saved.")

    # Evaluate
    print(f"\nRunning {CONFIG['n_folds']}-fold logistic regression ...")
    fold_results = run_kfold_evaluation(ref_labels, ref_groups, fold_assignments)
    agg = print_summary(fold_results)
    save_results(fold_results, agg)


if __name__ == "__main__":
    main()
