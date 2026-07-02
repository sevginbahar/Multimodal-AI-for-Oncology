"""
PanDerm K-Fold Evaluation
==========================
Reads patient-level features saved by extract_features.py and runs
logistic regression classification, then generates all plots.

Requires CPU only. Run after extract_features.py.

Usage:
    python evaluate.py

Outputs (saved to OUTPUT_DIR):
    kfold_results_finetuned.csv
    fold_{i}_report.txt
    confusion_matrix_aggregate.png
    roc_curves_mean.png
    fold_accuracy_bars.png
    umap_patient_features.png
    attention_map_examples_finetuned.png
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["finetune", "frozen"], default="finetune")
args, _ = parser.parse_known_args()
MODE_LABEL  = "Fine-Tuned" if args.mode == "finetune" else "Frozen"
RESULTS_CSV = f"kfold_results_{args.mode}.csv"
import torch
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
from collections import OrderedDict
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score,
    classification_report, confusion_matrix,
    roc_auc_score, cohen_kappa_score,
    roc_curve, auc,
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PANDERM_CLASS, CHECKPOINT_LARGE,
    OUTPUT_DIR, FEATURES_DIR,
    CLASS_NAMES, CLASS_LABELS, DISPLAY_NAMES,
    IMAGENET_MEAN, IMAGENET_STD,
    NB_CLASSES, N_FOLDS,
    LOGREG_C, LOGREG_MAX_ITER,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COLORS = ["#1565c0", "#d32f2f", "#6a1b9a"]


# ── Logistic regression evaluation ───────────────────────────────────────

def run_kfold_evaluation(patient_labels, group_ids, fold_assignments):
    fold_results = []

    for fold_idx in range(N_FOLDS):
        feat_path = FEATURES_DIR / f"patient_features_fold{fold_idx}.npy"
        if not feat_path.exists():
            print(f"  Fold {fold_idx}: features missing — run extract_features.py first")
            continue

        patient_features = np.load(str(feat_path))
        train_mask = np.array([fold_assignments[gid] != fold_idx for gid in group_ids])
        test_mask  = ~train_mask
        X_tr, y_tr = patient_features[train_mask], patient_labels[train_mask]
        X_te, y_te = patient_features[test_mask],  patient_labels[test_mask]

        clf = LogisticRegression(
            C=LOGREG_C, max_iter=LOGREG_MAX_ITER,
            class_weight="balanced", solver="lbfgs",
            random_state=42, multi_class="multinomial",
        )
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
            "y_test":  y_te,
            "y_proba": y_proba,
            "n_train": int(train_mask.sum()),
            "n_test":  int(test_mask.sum()),
            "report_text": classification_report(y_te, y_pred,
                                                  target_names=CLASS_NAMES, zero_division=0),
        }
        fold_results.append(result)
        print(f"  Fold {fold_idx+1}: BalAcc={result['balanced_accuracy']:.3f}  "
              f"AUC={macro_auc:.3f}  F1={result['macro_f1']:.3f}  "
              f"(train={result['n_train']}, test={result['n_test']})")

    return fold_results


def _ci95(vals):
    vals = np.array(vals)
    lo, hi = stats.t.interval(0.95, df=len(vals)-1,
                               loc=np.nanmean(vals), scale=stats.sem(vals, nan_policy="omit"))
    return lo, hi


def print_summary(fold_results):
    scalar_metrics = ["balanced_accuracy", "accuracy", "macro_auc", "cohen_kappa", "macro_f1"]
    agg = {}
    for m in scalar_metrics:
        vals = [r[m] for r in fold_results]
        agg[f"{m}_mean"] = np.nanmean(vals)
        agg[f"{m}_std"]  = np.nanstd(vals)
        agg[f"{m}_ci"]   = _ci95(vals)
    for cls in CLASS_NAMES:
        for mt in ["f1", "precision", "recall", "auc"]:
            vals = [r[f"per_class_{mt}"][cls] for r in fold_results]
            agg[f"{cls}_{mt}_mean"] = np.nanmean(vals)
            agg[f"{cls}_{mt}_std"]  = np.nanstd(vals)
            agg[f"{cls}_{mt}_ci"]   = _ci95(vals)

    print("\n" + "="*75)
    print(f"RESULTS — {MODE_LABEL} PanDerm (Patient-Level, 5-Fold CV)")
    print("="*75)
    for label, key in [("Balanced Accuracy", "balanced_accuracy"),
                        ("Accuracy",          "accuracy"),
                        ("Macro AUC",         "macro_auc"),
                        ("Macro F1",          "macro_f1"),
                        ("Cohen Kappa",       "cohen_kappa")]:
        lo, hi = agg[f"{key}_ci"]
        print(f"  {label:<20}: {agg[f'{key}_mean']:.3f} ± {agg[f'{key}_std']:.3f}"
              f"  (95% CI {lo:.3f}–{hi:.3f})")
    print(f"\n  {'Class':<22} {'F1 (95% CI)':>26} {'AUC (95% CI)':>26}")
    print("  " + "-"*76)
    for cls in CLASS_NAMES:
        f1_lo, f1_hi = agg[f"{cls}_f1_ci"]
        au_lo, au_hi = agg[f"{cls}_auc_ci"]
        f1 = f"{agg[f'{cls}_f1_mean']:.3f}±{agg[f'{cls}_f1_std']:.3f} [{f1_lo:.3f}–{f1_hi:.3f}]"
        au = f"{agg[f'{cls}_auc_mean']:.3f}±{agg[f'{cls}_auc_std']:.3f} [{au_lo:.3f}–{au_hi:.3f}]"
        print(f"  {DISPLAY_NAMES[cls]:<22} {f1:>26} {au:>26}")
    print("="*75)
    return agg


def save_results(fold_results, agg):
    rows = []
    for r in fold_results:
        row = {k: r[k] for k in ["fold", "balanced_accuracy", "accuracy",
                                   "macro_auc", "macro_f1", "cohen_kappa",
                                   "n_train", "n_test"]}
        for cls in CLASS_NAMES:
            row[f"{cls}_f1"]  = r["per_class_f1"][cls]
            row[f"{cls}_auc"] = r["per_class_auc"][cls]
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / RESULTS_CSV, index=False)
    for r in fold_results:
        (OUTPUT_DIR / f"fold_{r['fold']}_report.txt").write_text(
            f"Fold {r['fold']} — train={r['n_train']} test={r['n_test']}\n\n"
            + r["report_text"]
        )
    print(f"\nResults saved to: {OUTPUT_DIR}")


# ── Plots ─────────────────────────────────────────────────────────────────

def plot_confusion_matrix(fold_results: list, output_path: Path):
    agg_cm  = sum(r["confusion_matrix"] for r in fold_results)
    cm_norm = agg_cm.astype(float) / (agg_cm.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=False, cmap="Blues", vmin=0, vmax=1,
                xticklabels=[DISPLAY_NAMES[c] for c in CLASS_NAMES],
                yticklabels=[DISPLAY_NAMES[c] for c in CLASS_NAMES], ax=ax)
    for i in range(agg_cm.shape[0]):
        for j in range(agg_cm.shape[1]):
            ax.text(j+0.5, i+0.5,
                    f"{cm_norm[i,j]:.0%}\n(n={agg_cm[i,j]})",
                    ha="center", va="center", fontsize=11,
                    color="white" if cm_norm[i,j] > 0.5 else "black")
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Aggregate Confusion Matrix (Row-Normalised)", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path.name}")


def plot_roc_curves(fold_results: list, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 7))
    for ci, cls in enumerate(CLASS_NAMES):
        all_fpr     = np.linspace(0, 1, 100)
        tpr_interps = []
        for r in fold_results:
            y_bin = label_binarize(r["y_test"], classes=[0, 1, 2])
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, ci], r["y_proba"][:, ci])
                ti = np.interp(all_fpr, fpr, tpr); ti[0] = 0.0
                tpr_interps.append(ti)
            except ValueError:
                pass
        mean_tpr = np.mean(tpr_interps, axis=0); mean_tpr[-1] = 1.0
        std_tpr  = np.std(tpr_interps, axis=0)
        mean_auc = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, color=COLORS[ci], linewidth=2,
                label=f"{DISPLAY_NAMES[cls]} (AUC={mean_auc:.3f})")
        ax.fill_between(all_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                        color=COLORS[ci], alpha=0.15)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Mean ROC Curves (One-vs-Rest, 5-Fold CV)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path.name}")


def plot_fold_accuracy_bars(fold_results: list, output_path: Path):
    folds    = [r["fold"] + 1 for r in fold_results]
    bal_accs = [r["balanced_accuracy"] for r in fold_results]
    mean_acc = np.mean(bal_accs)
    fig, ax  = plt.subplots(figsize=(7, 4))
    bars = ax.bar(folds, bal_accs, color="#1976d2", alpha=0.8, edgecolor="white")
    ax.axhline(mean_acc, color="#d32f2f", linestyle="--", linewidth=2,
               label=f"Mean = {mean_acc:.3f}")
    ax.axhline(1 / len(CLASS_NAMES), color="gray", linestyle=":", linewidth=1.5,
               alpha=0.6, label="Random baseline")
    for bar, val in zip(bars, bal_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xlabel("Fold"); ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Balanced Accuracy per Fold", fontsize=13, fontweight="bold")
    ax.set_xticks(folds); ax.set_ylim(0, 1); ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path.name}")


def plot_umap(fold_results: list, output_path: Path):
    try:
        import umap as umap_lib
    except ImportError:
        print("  umap-learn not installed — skipping (pip install umap-learn)")
        return
    all_probs  = np.vstack([r["y_proba"] for r in fold_results])
    all_labels = np.concatenate([r["y_test"] for r in fold_results])
    print(f"  Fitting UMAP on {len(all_labels)} test patients...")
    embedding = umap_lib.UMAP(n_neighbors=15, min_dist=0.1,
                               n_components=2, random_state=42).fit_transform(all_probs)
    fig, ax = plt.subplots(figsize=(9, 7))
    for lv, cls in enumerate(CLASS_NAMES):
        mask = all_labels == lv
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=COLORS[lv], s=40, alpha=0.7, edgecolors="k", linewidth=0.5,
                   label=DISPLAY_NAMES[cls])
    ax.set_title(f"UMAP — {MODE_LABEL} PanDerm Features (5-Fold)", fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); ax.legend()
    ax.grid(linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path.name}")


# ── Attention maps (uses GPU if available) ────────────────────────────────

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
    model.load_state_dict(clean, strict=False)
    model.eval()
    return model.to(DEVICE)


def get_attention_map(model: torch.nn.Module, img_path: str) -> tuple:
    vis_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    pil_img    = Image.open(img_path).convert("RGB")
    img_tensor = vis_tf(pil_img).unsqueeze(0)
    attn_captured = {}

    def hook_fn(module, input, output):
        x = input[0]
        B, N, C = x.shape
        qkv = module.qkv(x).reshape(
            B, N, 3, module.num_heads, C // module.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn_captured["weights"] = attn.softmax(dim=-1).detach().cpu()

    handle = model.blocks[-1].attn.register_forward_hook(hook_fn)
    with torch.no_grad():
        model.forward_features(img_tensor.to(DEVICE), is_train=False)
    handle.remove()

    cls_attn  = attn_captured["weights"][0, :, 0, 1:]
    mean_attn = cls_attn.mean(dim=0)
    import numpy as np
    side      = int(np.sqrt(mean_attn.shape[0]))
    attn_map  = mean_attn.reshape(side, side).numpy()
    attn_map  = cv2.resize(attn_map, (224, 224), interpolation=cv2.INTER_CUBIC)

    resized     = np.array(pil_img.resize((224, 224)))
    gray        = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    lesion_mask = (gray > 10).astype(np.float32)
    attn_masked = attn_map * lesion_mask
    lesion_vals = attn_masked[lesion_mask == 1]

    if lesion_vals.size > 0 and lesion_vals.max() > lesion_vals.min():
        vmin, vmax = lesion_vals.min(), lesion_vals.max()
        attn_norm  = np.clip((attn_masked - vmin) / (vmax - vmin), 0, 1)
    else:
        attn_norm  = attn_masked

    heatmap_u8                = (attn_norm * 255).astype(np.uint8)
    heatmap                   = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap[lesion_mask == 0] = 0
    overlay                   = cv2.addWeighted(resized, 0.6, heatmap, 0.4, 0)
    return pil_img.resize((224, 224)), overlay


def generate_attention_maps(manifest: pd.DataFrame, model: torch.nn.Module,
                             output_dir: Path, n_per_class: int = 3):
    import random
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths, labels = [], []
    for cls in CLASS_NAMES:
        subset = manifest[manifest["diagnosis"] == cls]["input_image"].tolist()
        sample = random.sample(subset, min(n_per_class, len(subset)))
        for p in sample:
            image_paths.append(p); labels.append(CLASS_LABELS[cls])

    n = len(image_paths)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1: axes = np.array([axes])

    for i, (img_path, label) in enumerate(zip(image_paths, labels)):
        orig, overlay = get_attention_map(model, img_path)
        true_label    = DISPLAY_NAMES[CLASS_NAMES[label]]
        axes[i, 0].imshow(orig);    axes[i, 0].set_title(f"Original — {true_label}", fontsize=9)
        axes[i, 1].imshow(overlay); axes[i, 1].set_title("Attention Map", fontsize=9)
        for ax in axes[i]: ax.axis("off")
        print(f"  {os.path.basename(img_path)}")

    plt.suptitle(f"PanDerm Attention Maps ({MODE_LABEL})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = output_dir / f"attention_map_examples_{args.mode}.png"
    plt.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    # Load saved patient labels and fold assignments
    labels_path = FEATURES_DIR / "patient_labels.npy"
    groups_path = FEATURES_DIR / "patient_group_ids.npy"
    folds_path  = FEATURES_DIR / "group_fold_mapping.csv"

    if not all(p.exists() for p in [labels_path, groups_path, folds_path]):
        raise FileNotFoundError(
            "Features not found. Run extract_features.py first."
        )

    patient_labels  = np.load(str(labels_path))
    group_ids       = np.load(str(groups_path), allow_pickle=True).tolist()
    fold_df         = pd.read_csv(folds_path)
    fold_assignments = dict(zip(fold_df["group_id"], fold_df["fold"]))

    # Evaluate
    print(f"\nRunning {N_FOLDS}-fold logistic regression...")
    fold_results = run_kfold_evaluation(patient_labels, group_ids, fold_assignments)
    agg = print_summary(fold_results)
    save_results(fold_results, agg)

    # Plots
    print("\nGenerating plots...")
    plot_confusion_matrix(fold_results,   OUTPUT_DIR / "confusion_matrix_aggregate.png")
    plot_roc_curves(fold_results,         OUTPUT_DIR / "roc_curves_mean.png")
    plot_fold_accuracy_bars(fold_results, OUTPUT_DIR / "fold_accuracy_bars.png")
    plot_umap(fold_results,               OUTPUT_DIR / "umap_patient_features.png")

    # Attention maps — load fold 0 checkpoint
    print("\nGenerating attention maps (fold 0)...")
    try:
        manifest = pd.read_csv(OUTPUT_DIR / "dataset_manifest.csv")
        manifest["input_image"] = manifest["segmented_path"]
        ckpt_path  = find_best_checkpoint(OUTPUT_DIR / "results_fold0")
        attn_model = load_panderm_encoder(ckpt_path)
        generate_attention_maps(manifest, attn_model, OUTPUT_DIR, n_per_class=3)
        del attn_model; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Attention maps failed: {e}")


if __name__ == "__main__":
    main()
