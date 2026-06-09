"""
Late Fusion — PanDerm + BioClinicalBERT
========================================
Combines fine-tuned PanDerm image features (1024-dim) with BioClinicalBERT
text embeddings (768-dim) for multimodal patient-level classification of
melanocytic skin lesions.

Fusion strategy: concatenation -> logistic regression (5-fold CV)
Evaluated on 138/177 patients with both image and clinical text data.

Usage:
    python late_fusion.py

Outputs (saved to OUTPUT_DIR/fusion_results/):
    fusion_kfold_results.csv        — aggregate results
    fold_{i}_report.txt             — per-fold classification reports
    fusion_confusion_matrix.png     — aggregate confusion matrix
    fusion_roc_curves.png           — mean ROC curves
    fusion_umap.png                 — UMAP of fused feature space

Results:
    Balanced Accuracy : 0.603 +/- 0.060
    Macro AUC         : 0.778 +/- 0.057
    Macro F1          : 0.576 +/- 0.061
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score,
    classification_report, confusion_matrix,
    roc_auc_score, cohen_kappa_score,
    roc_curve, auc as sk_auc,
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OUTPUT_DIR, FEATURES_DIR, CLINICAL_DIR,
    CLASS_NAMES, CLASS_LABELS, DISPLAY_NAMES,
    N_FOLDS, LOGREG_C, LOGREG_MAX_ITER,
)

# ── Config ────────────────────────────────────────────────────────────────
COLORS = ["#1565c0", "#d32f2f", "#6a1b9a"]
OUT    = OUTPUT_DIR / "fusion_results"
OUT.mkdir(parents=True, exist_ok=True)


# ── Load and align modalities ─────────────────────────────────────────────
def load_and_align():
    # Clinical embeddings
    clin_emb = np.load(str(CLINICAL_DIR / "clinical_embeddings.npy"))          # (177, 768)
    clin_df  = pd.read_csv(str(CLINICAL_DIR / "full_reports.csv"))
    clin_df["patient_id"] = clin_df["clinical_history_number"].astype(str)
    clin_df["clin_idx"]   = range(len(clin_df))
    print(f"Clinical embeddings : {clin_emb.shape}")

    # Image patient metadata
    group_ids  = list(np.load(str(FEATURES_DIR / "patient_group_ids.npy"), allow_pickle=True))
    img_labels = np.load(str(FEATURES_DIR / "patient_labels.npy"))
    fold_df    = pd.read_csv(FEATURES_DIR / "group_fold_mapping.csv")
    fold_assign = dict(zip(fold_df["group_id"], fold_df["fold"]))

    img_df = pd.DataFrame({
        "group_id":   group_ids,
        "patient_id": [g.split("__")[0] for g in group_ids],
        "diagnosis":  [g.split("__")[1] for g in group_ids],
        "label":      img_labels,
        "fold":       [fold_assign[g] for g in group_ids],
        "img_idx":    range(len(group_ids)),
    })

    # Inner join on patient_id
    merged = img_df.merge(clin_df[["patient_id", "clin_idx"]], on="patient_id", how="inner")
    merged = merged.reset_index(drop=True)

    print(f"Image patients      : {len(img_df)}")
    print(f"Matched patients    : {len(merged)} (both modalities)")
    print(f"Class distribution  : {merged['diagnosis'].value_counts().to_dict()}")
    return merged, clin_emb


# ── Build fused features ──────────────────────────────────────────────────
def build_fused_features(merged, clin_emb):
    fused_features = {}
    for fold_idx in range(N_FOLDS):
        feat_path = FEATURES_DIR / f"patient_features_fold{fold_idx}.npy"
        if not feat_path.exists():
            print(f"Fold {fold_idx}: image features missing — run extract_features.py first")
            continue
        all_img = np.load(str(feat_path))                           # (166, 1024)
        img_f   = all_img[merged["img_idx"].values]                  # (138, 1024)
        txt_f   = clin_emb[merged["clin_idx"].values]                # (138,  768)
        fused   = np.concatenate([img_f, txt_f], axis=1)             # (138, 1792)
        fused_features[fold_idx] = fused
        print(f"Fold {fold_idx}: {img_f.shape} + {txt_f.shape} -> {fused.shape}")
    return fused_features


# ── K-fold evaluation ─────────────────────────────────────────────────────
def run_kfold_evaluation(fused_features, merged):
    labels        = merged["label"].values
    fold_assign_m = merged["fold"].values
    fold_results  = []

    for fold_idx in range(N_FOLDS):
        if fold_idx not in fused_features:
            continue
        fused      = fused_features[fold_idx]
        train_mask = fold_assign_m != fold_idx
        test_mask  = ~train_mask
        X_tr, y_tr = fused[train_mask], labels[train_mask]
        X_te, y_te = fused[test_mask],  labels[test_mask]

        clf = LogisticRegression(C=LOGREG_C, max_iter=LOGREG_MAX_ITER, class_weight="balanced",
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
            "y_test": y_te, "y_pred": y_pred, "y_proba": y_proba,
            "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
            "report_text": classification_report(y_te, y_pred,
                                                 target_names=CLASS_NAMES, zero_division=0),
        }
        fold_results.append(result)
        print(f"  Fold {fold_idx+1}: BalAcc={result['balanced_accuracy']:.3f}  "
              f"AUC={macro_auc:.3f}  F1={result['macro_f1']:.3f}  "
              f"(train={result['n_train']}, test={result['n_test']})")

    return fold_results


# ── Aggregate + print ─────────────────────────────────────────────────────
def aggregate_and_print(fold_results):
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
    agg_cm = sum(r["confusion_matrix"] for r in fold_results)

    print("\n" + "="*70)
    print("RESULTS -- Late Fusion: PanDerm + BioClinicalBERT (5-Fold CV)")
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
    return agg, agg_cm


# ── Save results ──────────────────────────────────────────────────────────
def save_results(fold_results):
    rows = []
    for r in fold_results:
        row = {k: r[k] for k in ["fold","balanced_accuracy","accuracy",
                                   "macro_auc","macro_f1","cohen_kappa","n_train","n_test"]}
        for cls in CLASS_NAMES:
            row[f"{cls}_f1"]  = r["per_class_f1"][cls]
            row[f"{cls}_auc"] = r["per_class_auc"][cls]
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "fusion_kfold_results.csv", index=False)
    for r in fold_results:
        (OUT / f"fold_{r['fold']}_report.txt").write_text(
            f"Fold {r['fold']} -- train={r['n_train']} test={r['n_test']}\n\n" + r["report_text"])
    print(f"Saved results to {OUT}")


# ── Plots ─────────────────────────────────────────────────────────────────
def plot_confusion_matrix(agg_cm):
    cm_norm = agg_cm.astype(float) / (agg_cm.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(agg_cm, annot=False, cmap="Blues",
                xticklabels=[DISPLAY_NAMES[c] for c in CLASS_NAMES],
                yticklabels=[DISPLAY_NAMES[c] for c in CLASS_NAMES], ax=ax)
    for i in range(agg_cm.shape[0]):
        for j in range(agg_cm.shape[1]):
            ax.text(j+0.5, i+0.5, f"{agg_cm[i,j]}\n({cm_norm[i,j]:.0%})",
                    ha="center", va="center", fontsize=11,
                    color="white" if cm_norm[i,j] > 0.5 else "black")
    ax.set_title("Fusion — Aggregate Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(str(OUT / "fusion_confusion_matrix.png"), dpi=200, bbox_inches="tight")
    print("Saved: fusion_confusion_matrix.png")


def plot_roc_curves(fold_results):
    fig, ax = plt.subplots(figsize=(8, 7))
    for ci, cls in enumerate(CLASS_NAMES):
        all_fpr = np.linspace(0, 1, 100)
        tpr_interps = []
        for r in fold_results:
            y_bin = label_binarize(r["y_test"], classes=[0,1,2])
            fpr, tpr, _ = roc_curve(y_bin[:, ci], r["y_proba"][:, ci])
            ti = np.interp(all_fpr, fpr, tpr); ti[0] = 0.0
            tpr_interps.append(ti)
        mean_tpr = np.mean(tpr_interps, axis=0); mean_tpr[-1] = 1.0
        std_tpr  = np.std(tpr_interps, axis=0)
        ax.plot(all_fpr, mean_tpr, color=COLORS[ci], linewidth=2,
                label=f"{DISPLAY_NAMES[cls]} (AUC={sk_auc(all_fpr, mean_tpr):.3f})")
        ax.fill_between(all_fpr, mean_tpr-std_tpr, mean_tpr+std_tpr,
                        color=COLORS[ci], alpha=0.15)
    ax.plot([0,1],[0,1],"k--",linewidth=1,alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Fusion — Mean ROC Curves (One-vs-Rest, 5-Fold CV)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(str(OUT / "fusion_roc_curves.png"), dpi=200, bbox_inches="tight")
    print("Saved: fusion_roc_curves.png")


def plot_umap(fused_features, labels):
    try:
        import umap
        reducer   = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding = reducer.fit_transform(fused_features[0])
        fig, ax   = plt.subplots(figsize=(9, 7))
        for lv, cls in enumerate(CLASS_NAMES):
            mask = labels == lv
            ax.scatter(embedding[mask,0], embedding[mask,1],
                       c=COLORS[lv], s=40, alpha=0.7, edgecolors="k", linewidth=0.5,
                       label=DISPLAY_NAMES[cls])
        ax.set_title("UMAP — Fused Features (PanDerm + BioClinicalBERT)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); ax.legend()
        ax.grid(linestyle="--", alpha=0.3); plt.tight_layout()
        plt.savefig(str(OUT / "fusion_umap.png"), dpi=200, bbox_inches="tight")
        print("Saved: fusion_umap.png")
    except ImportError:
        print("umap-learn not installed — pip install umap-learn")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    merged, clin_emb = load_and_align()
    fused_features   = build_fused_features(merged, clin_emb)

    print(f"\nRunning {N_FOLDS}-fold logistic regression ...")
    fold_results     = run_kfold_evaluation(fused_features, merged)
    agg, agg_cm      = aggregate_and_print(fold_results)
    save_results(fold_results)

    plot_confusion_matrix(agg_cm)
    plot_roc_curves(fold_results)
    plot_umap(fused_features, merged["label"].values)

    print(f"\nDone. All outputs saved to {OUT}")


if __name__ == "__main__":
    main()
