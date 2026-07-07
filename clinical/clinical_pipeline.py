"""
Clinical Text Pipeline
======================
Extracts BioClinicalBERT embeddings from melanocytic skin lesion pathology reports.

Usage:
    python clinical_pipeline.py

Outputs:
    clinical_embeddings.npy   (177, 768)  — dense text embeddings
    clinical_features.csv     (177, 6)    — boolean clinical flags
    full_reports.csv          (177, 3)    — raw report text
    umap_embeddings.png                   — UMAP validation plot
"""

import sys
import re
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, roc_auc_score, cohen_kappa_score,
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_ROOT, CLINICAL_DIR, CLINICAL_INPUT,
    CLASS_NAMES, CLASS_LABELS, DISPLAY_NAMES,
)

# ── Config ────────────────────────────────────────────────────────────────
CONFIG = {
    "input_path":       str(CLINICAL_INPUT),
    "patient_id_col":   "clinical_history_number",
    "text_cols": {
        "diagnosis":    "diagnostic_summary_english",
        "macroscopic":  "macroscopic_description_english",
    },
    "label_col":        "source_diagnosis",
    "embedding_model":  "emilyalsentzer/Bio_ClinicalBERT",
    "max_token_length": 512,
    "batch_size":       16,
    "output_dir":       str(CLINICAL_DIR),
}



# ── Data loading ──────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".xlsx":
        df = pd.read_excel(path)
    else:
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")
            print("Note: file read with latin-1 encoding")
    print(f"Loaded {len(df)} rows x {len(df.columns)} columns")
    return df


# ── Diagnosis term stripping ──────────────────────────────────────────────
_STRIP_PATTERNS = [
    r"melanoma in situ",
    r"melanoma stage ia",
    r"malignant melanoma",
    r"superficial melanoma",
    r"melanoma",
    r"dysplastic nevus",
    r"melanocytic nevus",
    r"compound nevus",
    r"junctional nevus",
    r"nevus",
    r"clark level\s*:?\s*[ivxIVX0-9]+",
    r"breslow[^.]*?mm",
    r"breslow",
    r"pT[0-9is][abc]?",
    r"ptis",
    r"pathological stage[^.]*",
    r"stage i[ab]?",
    r"radial growth phase",
    r"dysplastic changes",
    r"dysplastic features",
    r"dysplastic",
    r"displastic[^.]*",
    r"peritumoral",
    r"papillary dermis invasion",
    r"I\.T\s*[0-9]+[A-Z]?",
]

def strip_diagnosis_terms(text: str) -> str:
    if not text:
        return ""
    for p in _STRIP_PATTERNS:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip()


# ── Report building ───────────────────────────────────────────────────────
def build_full_report(row: pd.Series) -> str:
    diag  = str(row.get(CONFIG["text_cols"]["diagnosis"],   "") or "").strip()
    macro = str(row.get(CONFIG["text_cols"]["macroscopic"], "") or "").strip()
    raw   = "\n".join(p for p in [diag, macro] if p).strip()
    return strip_diagnosis_terms(raw)



# ── BioClinicalBERT embeddings ────────────────────────────────────────────
def get_embeddings(
    texts: list,
    model_name: str,
    max_len: int,
    device: str,
    batch_size: int = 16,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch   = texts[i : i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_len, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output = model(**encoded)
        mask       = encoded["attention_mask"].unsqueeze(-1).float()
        summed     = (output.last_hidden_state * mask).sum(dim=1)
        counts     = mask.sum(dim=1).clamp(min=1e-9)
        embeddings = (summed / counts).cpu().numpy()
        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)


# ── Validation ────────────────────────────────────────────────────────────
def validate_embeddings(embeddings: np.ndarray, labels: np.ndarray, label_names: list,
                        texts: list = None, report_lengths: np.ndarray = None):
    print("\n── Validation ───────────────────────────────────────")

    # ── 1. Cosine similarity ──────────────────────────────────────────────
    sim_matrix = cosine_similarity(embeddings)
    within, between = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                within.append(sim_matrix[i, j])
            else:
                between.append(sim_matrix[i, j])

    print(f"\nCosine similarity:")
    print(f"  Within-class  : {np.mean(within):.4f} ± {np.std(within):.4f}")
    print(f"  Between-class : {np.mean(between):.4f} ± {np.std(between):.4f}")
    print(f"  Gap           : {np.mean(within) - np.mean(between):.4f}  (positive = clinically meaningful)")

    # ── 2. k-NN LOO accuracy ─────────────────────────────────────────────
    print("\nRunning k-NN leave-one-out ...")
    loo   = LeaveOneOut()
    knn   = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    preds = []
    for train_idx, test_idx in loo.split(embeddings):
        knn.fit(embeddings[train_idx], labels[train_idx])
        preds.append(knn.predict(embeddings[test_idx])[0])

    acc = accuracy_score(labels, preds)
    print(f"k-NN LOO accuracy : {acc:.3f} ({acc*100:.1f}%)")
    print(f"Random baseline   : {1/len(label_names):.3f} ({100/len(label_names):.1f}%)")

    # ── 3. Permutation test ───────────────────────────────────────────────
    print("\nRunning permutation test (100 shuffles) ...")
    rng      = np.random.default_rng(42)
    perm_accs = []
    for _ in range(100):
        shuffled = rng.permutation(labels)
        p_preds  = []
        for train_idx, test_idx in loo.split(embeddings):
            knn.fit(embeddings[train_idx], shuffled[train_idx])
            p_preds.append(knn.predict(embeddings[test_idx])[0])
        perm_accs.append(accuracy_score(shuffled, p_preds))
    perm_mean = np.mean(perm_accs)
    p_value   = np.mean(np.array(perm_accs) >= acc)
    print(f"  Real accuracy     : {acc:.3f}")
    print(f"  Permuted mean     : {perm_mean:.3f} ± {np.std(perm_accs):.3f}")
    print(f"  p-value           : {p_value:.3f}  {'✅ significant' if p_value < 0.05 else '⚠️ not significant'}")

    # ── 4. Report length confound ─────────────────────────────────────────
    if report_lengths is not None:
        print("\nReport length per class (characters):")
        for i, name in enumerate(label_names):
            mask   = labels == i
            lengths = report_lengths[mask]
            print(f"  {name:<12} : {np.mean(lengths):6.0f} ± {np.std(lengths):.0f}  "
                  f"(min={lengths.min():.0f}, max={lengths.max():.0f})")
        # Spearman correlation between length and label
        from scipy.stats import spearmanr
        corr, pval = spearmanr(report_lengths, labels)
        print(f"  Length-label correlation (Spearman r={corr:.3f}, p={pval:.3f})"
              f"  {'⚠️ length is a confound' if pval < 0.05 else '✅ length not a confound'}")

    # ── 5. Flag-only baseline ─────────────────────────────────────────────
    # (only runs if called from main where flags are available — skipped otherwise)

    # ── 6. Per-class k-NN breakdown ───────────────────────────────────────
    from sklearn.metrics import classification_report
    print("\nPer-class k-NN breakdown:")
    print(classification_report(labels, preds, target_names=label_names,
                                zero_division=0))

    # UMAP
    try:
        import umap
        import matplotlib.pyplot as plt

        print("\nRunning UMAP ...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        emb_2d  = reducer.fit_transform(embeddings)

        colours = plt.cm.Set1(np.linspace(0, 1, len(label_names)))
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, label in enumerate(label_names):
            mask = labels == i
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                       label=label, color=colours[i], alpha=0.7, s=40)
        ax.legend(fontsize=9)
        ax.set_title("UMAP — BioClinicalBERT embeddings")
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        plt.tight_layout()
        plt.savefig(Path(CONFIG["output_dir"]) / "umap_embeddings.png", dpi=150)
        plt.show()
        print("Saved: umap_embeddings.png")
    except ImportError:
        print("umap-learn not installed — skipping UMAP (pip install umap-learn)")


# ── LR 5-fold CV (matched protocol) ──────────────────────────────────────
def run_lr_5fold(embeddings: np.ndarray, labels: np.ndarray,
                 label_names: list, out_dir: Path):
    skf          = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(embeddings, labels)):
        X_tr, y_tr = embeddings[train_idx], labels[train_idx]
        X_te, y_te = embeddings[test_idx],  labels[test_idx]

        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                                 solver="lbfgs", random_state=42,
                                 multi_class="multinomial")
        clf.fit(X_tr, y_tr)
        y_pred  = clf.predict(X_te)
        y_proba = clf.predict_proba(X_te)

        y_bin  = label_binarize(y_te, classes=list(range(len(label_names))))
        report = classification_report(y_te, y_pred, target_names=label_names,
                                       output_dict=True, zero_division=0)
        try:    macro_auc = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
        except: macro_auc = float("nan")

        per_class_auc = {}
        for ci, cls in enumerate(label_names):
            try:    per_class_auc[cls] = roc_auc_score(y_bin[:, ci], y_proba[:, ci])
            except: per_class_auc[cls] = float("nan")

        fold_results.append({
            "fold":              fold_idx,
            "balanced_accuracy": balanced_accuracy_score(y_te, y_pred),
            "accuracy":          accuracy_score(y_te, y_pred),
            "macro_auc":         macro_auc,
            "macro_f1":          report["macro avg"]["f1-score"],
            "cohen_kappa":       cohen_kappa_score(y_te, y_pred),
            "per_class_f1":      {c: report[c]["f1-score"] for c in label_names},
            "per_class_auc":     per_class_auc,
            "n_train":           len(train_idx),
            "n_test":            len(test_idx),
        })
        print(f"  Fold {fold_idx+1}: BalAcc={fold_results[-1]['balanced_accuracy']:.3f}  "
              f"AUC={macro_auc:.3f}  F1={fold_results[-1]['macro_f1']:.3f}")

    # Aggregate
    scalar_metrics = ["balanced_accuracy", "accuracy", "macro_auc", "macro_f1", "cohen_kappa"]
    agg = {}
    for m in scalar_metrics:
        vals = [r[m] for r in fold_results]
        agg[f"{m}_mean"] = np.nanmean(vals)
        agg[f"{m}_std"]  = np.nanstd(vals)
        lo, hi = stats.t.interval(0.95, df=len(vals)-1,
                                  loc=np.nanmean(vals), scale=stats.sem(vals))
        agg[f"{m}_ci"] = (lo, hi)

    print("\n" + "="*75)
    print("RESULTS — BioClinicalBERT LR (5-Fold CV, matched protocol)")
    print("="*75)
    for label, key in [("Balanced Accuracy", "balanced_accuracy"),
                        ("Accuracy",          "accuracy"),
                        ("Macro AUC",         "macro_auc"),
                        ("Macro F1",          "macro_f1"),
                        ("Cohen Kappa",       "cohen_kappa")]:
        lo, hi = agg[f"{key}_ci"]
        print(f"  {label:<20}: {agg[f'{key}_mean']:.3f} ± {agg[f'{key}_std']:.3f}"
              f"  (95% CI {lo:.3f}–{hi:.3f})")
    print("="*75)

    # Save
    rows = []
    for r in fold_results:
        row = {k: r[k] for k in ["fold", "balanced_accuracy", "accuracy",
                                   "macro_auc", "macro_f1", "cohen_kappa",
                                   "n_train", "n_test"]}
        for cls in label_names:
            row[f"{cls}_f1"]  = r["per_class_f1"][cls]
            row[f"{cls}_auc"] = r["per_class_auc"][cls]
        rows.append(row)
    csv_path = out_dir / "text_only_kfold_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    return fold_results, agg


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    df = load_data(CONFIG["input_path"])

    # Harmonise patient ID — strip whitespace so merges are consistent
    df[CONFIG["patient_id_col"]] = df[CONFIG["patient_id_col"]].astype(str).str.strip()

    # Build reports
    df["full_report"] = df.apply(build_full_report, axis=1)
    lengths = df["full_report"].str.len()
    print(f"\nReports built : {len(df)}")
    print(f"Avg length    : {lengths.mean():.0f} chars")
    print(f"Empty reports : {(lengths == 0).sum()}")

    # Embeddings
    print("\nLoading BioClinicalBERT ...")
    embeddings = get_embeddings(
        texts      = df["full_report"].tolist(),
        model_name = CONFIG["embedding_model"],
        max_len    = CONFIG["max_token_length"],
        device     = device,
        batch_size = CONFIG["batch_size"],
    )
    print(f"\nEmbedding matrix shape : {embeddings.shape}")
    print(f"dtype                  : {embeddings.dtype}")

    # Validate
    le           = LabelEncoder()
    labels_enc   = le.fit_transform(df[CONFIG["label_col"]].values)
    report_lengths = df["full_report"].str.len().values.astype(float)

    validate_embeddings(
        embeddings     = embeddings,
        labels         = labels_enc,
        label_names    = list(le.classes_),
        texts          = df["full_report"].tolist(),
        report_lengths = report_lengths,
    )

    # LR 5-fold CV
    print("\n── LR 5-Fold CV (matched protocol) ─────────────────────────")
    run_lr_5fold(embeddings, labels_enc, list(le.classes_), out_dir)

    # Save
    emb_path = out_dir / "clinical_embeddings.npy"
    np.save(str(emb_path), embeddings)
    print(f"\nSaved: {emb_path}  shape={embeddings.shape}")

    df[[CONFIG["patient_id_col"], CONFIG["label_col"], "full_report"]].to_csv(
        out_dir / "full_reports.csv", index=False
    )
    print(f"Saved: full_reports.csv")

    print(f"\nDone — all outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
