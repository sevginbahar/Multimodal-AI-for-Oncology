"""
Late Fusion — PanDerm + BioClinicalBERT  (Strategy C)
======================================================
Per-modality PCA → concat → logistic regression (5-fold CV).

Fusion architecture:
    Image  (1024-dim) → StandardScaler → PCA(8) ─┐
    Text   ( 768-dim) → StandardScaler → PCA(8) ─┴→ concat(16) → LR

C is selected by nested cross-validation inside each outer fold.
All transforms are fit on the training split only — no leakage.

Usage:
    python late_fusion.py [--mode frozen|finetune]

Outputs (saved to OUTPUT_DIR/fusion_results/):
    fusion_kfold_results.csv        — per-fold metrics
    fold_{i}_report.txt             — classification reports
    fusion_confusion_matrix.png     — aggregate confusion matrix
    fusion_roc_curves.png           — mean ROC curves (per-fold interpolated)
    fusion_umap.png                 — UMAP of PCA-fused features
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score,
    classification_report, confusion_matrix,
    roc_auc_score, cohen_kappa_score,
    roc_curve, auc as sk_auc,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OUTPUT_DIR, FEATURES_DIR, CLINICAL_DIR,
    PIPELINE_DIR,
    CLASS_NAMES, CLASS_LABELS, DISPLAY_NAMES,
    N_FOLDS, LOGREG_MAX_ITER, RANDOM_SEED,
)

# ── Fusion hyperparameters ────────────────────────────────────────────────
PCA_IMG_DIM  = 8                             # image PCA dims
PCA_TXT_DIM  = 8                             # text PCA dims
C_GRID       = [0.01, 0.03, 0.1, 0.3, 1.0]  # nested CV grid

COLORS = ["#1565c0", "#d32f2f", "#6a1b9a"]

# ── Parse mode ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["finetune", "frozen"], default="finetune")
args, _ = parser.parse_known_args()

if args.mode == "frozen":
    _IMG_FEATURES_DIR = FEATURES_DIR
    _IMG_OUTPUT_DIR   = OUTPUT_DIR
    _MODE_LABEL       = "Frozen PanDerm + BioClinicalBERT"
else:
    _IMG_FEATURES_DIR = FEATURES_DIR
    _IMG_OUTPUT_DIR   = OUTPUT_DIR
    _MODE_LABEL       = "Fine-Tuned PanDerm + BioClinicalBERT"

OUT = _IMG_OUTPUT_DIR / "fusion_results"
OUT.mkdir(parents=True, exist_ok=True)
print(f"Mode           : {_MODE_LABEL}")
print(f"Image features : {_IMG_FEATURES_DIR}")
print(f"Output dir     : {OUT}")
print(f"Fusion dims    : img={PCA_IMG_DIM}  txt={PCA_TXT_DIM}  total={PCA_IMG_DIM+PCA_TXT_DIM}")


# ── Load and align modalities ─────────────────────────────────────────────
def load_and_align():
    clin_emb = np.load(str(CLINICAL_DIR / "clinical_embeddings.npy"))
    clin_df  = pd.read_csv(str(CLINICAL_DIR / "full_reports.csv"))
    clin_df["patient_id"] = clin_df["clinical_history_number"].astype(str).str.strip()
    clin_df["clin_idx"]   = range(len(clin_df))
    print(f"Clinical embeddings : {clin_emb.shape}")

    group_ids  = list(np.load(str(_IMG_FEATURES_DIR / "patient_group_ids.npy"), allow_pickle=True))
    img_labels = np.load(str(_IMG_FEATURES_DIR / "patient_labels.npy"))
    fold_df    = pd.read_csv(_IMG_FEATURES_DIR / "group_fold_mapping.csv")
    fold_assign = dict(zip(fold_df["group_id"], fold_df["fold"]))

    img_df = pd.DataFrame({
        "group_id":   group_ids,
        "patient_id": [g.split("__")[0].strip() for g in group_ids],
        "diagnosis":  [g.split("__")[1] for g in group_ids],
        "label":      img_labels,
        "fold":       [fold_assign[g] for g in group_ids],
        "img_idx":    range(len(group_ids)),
    })

    merged = img_df.merge(clin_df[["patient_id", "clin_idx"]], on="patient_id", how="inner")
    merged = merged.reset_index(drop=True)

    print(f"Image patients      : {len(img_df)}")
    print(f"Matched patients    : {len(merged)} (both modalities)")
    print(f"Class distribution  : {merged['diagnosis'].value_counts().to_dict()}")
    return merged, clin_emb


# ── Per-modality PCA features (Strategy C) ───────────────────────────────
def get_pca_features(img_tr, img_te, txt_tr, txt_te):
    """StandardScaler → PCA per modality → concat. Fit on train only."""
    compressed_tr, compressed_te = [], []
    for X_tr, X_te, n_dim in [(img_tr, img_te, PCA_IMG_DIM),
                               (txt_tr, txt_te, PCA_TXT_DIM)]:
        sc  = StandardScaler()
        pca = PCA(n_components=min(n_dim, X_tr.shape[0]-1, X_tr.shape[1]),
                  random_state=RANDOM_SEED)
        compressed_tr.append(pca.fit_transform(sc.fit_transform(X_tr)))
        compressed_te.append(pca.transform(sc.transform(X_te)))
    return (np.concatenate(compressed_tr, axis=1),
            np.concatenate(compressed_te, axis=1))


# ── Nested CV for C selection ─────────────────────────────────────────────
def select_c(X_tr, y_tr):
    """Inner 3-fold stratified CV to pick best C from C_GRID."""
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    gs = GridSearchCV(
        LogisticRegression(max_iter=LOGREG_MAX_ITER, class_weight="balanced",
                           solver="lbfgs", random_state=RANDOM_SEED),
        param_grid={"C": C_GRID},
        cv=inner_cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    gs.fit(X_tr, y_tr)
    return gs.best_params_["C"]


# ── K-fold evaluation ─────────────────────────────────────────────────────
def run_kfold_evaluation(merged, clin_emb):
    labels        = merged["label"].values
    fold_assign_m = merged["fold"].values
    fold_results  = []

    for fold_idx in range(N_FOLDS):
        feat_path = _IMG_FEATURES_DIR / f"patient_features_fold{fold_idx}.npy"
        if not feat_path.exists():
            print(f"Fold {fold_idx}: features missing — run extract_features.py first")
            continue

        all_img = np.load(str(feat_path))
        img_f   = all_img[merged["img_idx"].values]
        txt_f   = clin_emb[merged["clin_idx"].values]

        train_mask = fold_assign_m != fold_idx
        test_mask  = ~train_mask

        X_tr, X_te = get_pca_features(
            img_f[train_mask], img_f[test_mask],
            txt_f[train_mask], txt_f[test_mask],
        )
        y_tr = labels[train_mask]
        y_te = labels[test_mask]

        best_c = select_c(X_tr, y_tr)

        clf = LogisticRegression(C=best_c, max_iter=LOGREG_MAX_ITER,
                                 class_weight="balanced", solver="lbfgs",
                                 random_state=RANDOM_SEED)
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
            "best_c":              best_c,
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
              f"C={best_c}  (train={result['n_train']}, test={result['n_test']})")

    return fold_results


# ── Aggregate + print ─────────────────────────────────────────────────────
def aggregate_and_print(fold_results):
    scalar_metrics = ["balanced_accuracy", "accuracy", "macro_auc", "cohen_kappa", "macro_f1"]
    agg = {}
    for m in scalar_metrics:
        vals = [r[m] for r in fold_results]
        lo, hi = stats.t.interval(0.95, df=len(vals)-1,
                                  loc=np.nanmean(vals), scale=stats.sem(vals))
        agg[f"{m}_mean"]  = np.nanmean(vals)
        agg[f"{m}_std"]   = np.nanstd(vals)
        agg[f"{m}_ci_lo"] = lo
        agg[f"{m}_ci_hi"] = hi

    for cls in CLASS_NAMES:
        for mt in ["f1", "precision", "recall", "auc"]:
            vals = [r[f"per_class_{mt}"][cls] for r in fold_results]
            agg[f"{cls}_{mt}_mean"] = np.nanmean(vals)
            agg[f"{cls}_{mt}_std"]  = np.nanstd(vals)

    agg_cm = sum(r["confusion_matrix"] for r in fold_results)
    c_vals = [r["best_c"] for r in fold_results]

    print("\n" + "="*70)
    print(f"RESULTS — Late Fusion: {_MODE_LABEL} (Strategy C, 5-Fold CV)")
    print(f"Dims: img={PCA_IMG_DIM} + txt={PCA_TXT_DIM} = {PCA_IMG_DIM+PCA_TXT_DIM} total")
    print(f"Best C per fold: {c_vals}  (nested CV)")
    print("="*70)
    for m in scalar_metrics:
        print(f"  {m:<22}: {agg[f'{m}_mean']:.3f} +/- {agg[f'{m}_std']:.3f}"
              f"  (95% CI {agg[f'{m}_ci_lo']:.3f}–{agg[f'{m}_ci_hi']:.3f})")
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
        row = {k: r[k] for k in ["fold", "best_c", "balanced_accuracy", "accuracy",
                                   "macro_auc", "macro_f1", "cohen_kappa", "n_train", "n_test"]}
        for cls in CLASS_NAMES:
            row[f"{cls}_f1"]  = r["per_class_f1"][cls]
            row[f"{cls}_auc"] = r["per_class_auc"][cls]
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "fusion_kfold_results.csv", index=False)
    for r in fold_results:
        (OUT / f"fold_{r['fold']}_report.txt").write_text(
            f"Fold {r['fold']} — train={r['n_train']} test={r['n_test']} best_C={r['best_c']}\n\n"
            + r["report_text"])
    print(f"Saved results to {OUT}")


# ── Plots ─────────────────────────────────────────────────────────────────
def plot_confusion_matrix(agg_cm):
    cm_norm = agg_cm.astype(float) / (agg_cm.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=False, cmap="Blues", vmin=0, vmax=1,
                xticklabels=[DISPLAY_NAMES[c] for c in CLASS_NAMES],
                yticklabels=[DISPLAY_NAMES[c] for c in CLASS_NAMES], ax=ax,
                cbar_kws={"format": lambda x, _: f"{x:.0%}"})
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j+0.5, i+0.5, f"{cm_norm[i,j]:.0%}\n(n={agg_cm[i,j]})",
                    ha="center", va="center", fontsize=11,
                    color="white" if cm_norm[i,j] > 0.5 else "black")
    ax.set_title("Fusion — Aggregate Confusion Matrix (Strategy C)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(str(OUT / "fusion_confusion_matrix.png"), dpi=200, bbox_inches="tight")
    print("Saved: fusion_confusion_matrix.png")


def plot_roc_curves(fold_results):
    fold_sizes = [r["n_test"] for r in fold_results]
    all_true   = np.concatenate([r["y_test"]  for r in fold_results])
    all_probs  = np.vstack([r["y_proba"] for r in fold_results])

    fig, ax = plt.subplots(figsize=(8, 7))
    ptr = 0
    for ci, cls in enumerate(CLASS_NAMES):
        all_fpr     = np.linspace(0, 1, 100)
        tpr_interps = []
        for r in fold_results:
            y_bin = label_binarize(r["y_test"], classes=[0, 1, 2])
            fpr, tpr, _ = roc_curve(y_bin[:, ci], r["y_proba"][:, ci])
            ti = np.interp(all_fpr, fpr, tpr); ti[0] = 0.0
            tpr_interps.append(ti)
        mean_tpr = np.mean(tpr_interps, axis=0); mean_tpr[-1] = 1.0
        std_tpr  = np.std(tpr_interps, axis=0)
        ax.plot(all_fpr, mean_tpr, color=COLORS[ci], linewidth=2,
                label=f"{DISPLAY_NAMES[cls]} (AUC={sk_auc(all_fpr, mean_tpr):.3f})")
        ax.fill_between(all_fpr, mean_tpr-std_tpr, mean_tpr+std_tpr,
                        color=COLORS[ci], alpha=0.15)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Fusion — Mean ROC Curves (One-vs-Rest, 5-Fold CV)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(str(OUT / "fusion_roc_curves.png"), dpi=200, bbox_inches="tight")
    print("Saved: fusion_roc_curves.png")


def plot_umap(merged, clin_emb):
    try:
        import umap
        all_img = np.load(str(_IMG_FEATURES_DIR / "patient_features_fold0.npy"))
        img_f   = all_img[merged["img_idx"].values]
        txt_f   = clin_emb[merged["clin_idx"].values]

        # Apply per-modality PCA on all patients (visualisation only — not fold-specific)
        compressed = []
        for X, n_dim in [(img_f, PCA_IMG_DIM), (txt_f, PCA_TXT_DIM)]:
            sc  = StandardScaler()
            pca = PCA(n_components=n_dim, random_state=RANDOM_SEED)
            compressed.append(pca.fit_transform(sc.fit_transform(X)))
        X_fused = np.concatenate(compressed, axis=1)

        # Pre-reduce before UMAP for stability
        pca_pre   = PCA(n_components=10, random_state=RANDOM_SEED)
        X_pre     = pca_pre.fit_transform(X_fused)
        reducer   = umap.UMAP(n_neighbors=30, min_dist=0.3, n_components=2,
                               metric="cosine", random_state=RANDOM_SEED)
        embedding = reducer.fit_transform(X_pre)
        labels    = merged["label"].values

        fig, ax = plt.subplots(figsize=(9, 7))
        for lv, cls in enumerate(CLASS_NAMES):
            mask = labels == lv
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=COLORS[lv], s=40, alpha=0.7, edgecolors="k", linewidth=0.5,
                       label=DISPLAY_NAMES[cls])
        ax.set_title("UMAP — Bimodal Fused Features (Strategy C, PCA-reduced)",
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
    fold_results     = run_kfold_evaluation(merged, clin_emb)
    agg, agg_cm      = aggregate_and_print(fold_results)
    save_results(fold_results)
    plot_confusion_matrix(agg_cm)
    plot_roc_curves(fold_results)
    plot_umap(merged, clin_emb)
    print(f"\nDone. All outputs saved to {OUT}")


if __name__ == "__main__":
    main()
