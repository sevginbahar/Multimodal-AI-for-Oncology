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

import re
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# ── Config ────────────────────────────────────────────────────────────────
CONFIG = {
    "input_path":       "/path/to/your/input",
    "patient_id_col":   "clinical_history_number",
    "text_cols": {
        "diagnosis":    "diagnostic_summary_english",
        "macroscopic":  "macroscopic_description_english",
    },
    "label_col":        "source_diagnosis",
    "embedding_model":  "emilyalsentzer/Bio_ClinicalBERT",
    "max_token_length": 512,
    "batch_size":       16,
    "output_dir":       "./outputs",
}

REGEX_FLAGS = {
    "flag_complete_excision": ["complete excision", "free margins", "free of lesion"],
    "flag_ulceration":        ["ulceration", "ulcerated"],
    "flag_mitosis":           ["mitosis", "mitotic"],
    "flag_vascular_invasion": ["vascular invasion", "lymphatic invasion", "lymphovascular"],
    "flag_regression":        ["regressive", "regression", "involutive"],
    "flag_nevus_component":   ["melanocytic nevus", "compound nevus", "junctional nevus"],
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


# ── Report building ───────────────────────────────────────────────────────
def build_full_report(row: pd.Series) -> str:
    diag  = str(row.get(CONFIG["text_cols"]["diagnosis"],   "") or "").strip()
    macro = str(row.get(CONFIG["text_cols"]["macroscopic"], "") or "").strip()
    return "\n".join(p for p in [diag, macro] if p).strip()


# ── Boolean flags ─────────────────────────────────────────────────────────
def extract_flag(text: str, keywords: list) -> bool:
    if pd.isna(text):
        return False
    return any(k in text.lower() for k in keywords)


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
def validate_embeddings(embeddings: np.ndarray, labels: np.ndarray, label_names: list):
    print("\n── Validation ───────────────────────────────────────")

    # Cosine similarity
    sim_matrix = cosine_similarity(embeddings)
    within, between = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                within.append(sim_matrix[i, j])
            else:
                between.append(sim_matrix[i, j])

    print(f"Cosine similarity:")
    print(f"  Within-class  : {np.mean(within):.4f} ± {np.std(within):.4f}")
    print(f"  Between-class : {np.mean(between):.4f} ± {np.std(between):.4f}")
    print(f"  Gap           : {np.mean(within) - np.mean(between):.4f}")

    # k-NN LOO accuracy
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


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    df = load_data(CONFIG["input_path"])

    # Build reports
    df["full_report"] = df.apply(build_full_report, axis=1)
    lengths = df["full_report"].str.len()
    print(f"\nReports built : {len(df)}")
    print(f"Avg length    : {lengths.mean():.0f} chars")
    print(f"Empty reports : {(lengths == 0).sum()}")

    # Boolean flags
    for flag, keywords in REGEX_FLAGS.items():
        df[flag] = df["full_report"].apply(lambda t: extract_flag(t, keywords))
    flag_cols = list(REGEX_FLAGS.keys())
    print("\nBoolean flag totals:")
    print(df[flag_cols].sum().to_string())

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
    le         = LabelEncoder()
    labels_enc = le.fit_transform(df[CONFIG["label_col"]].values)
    validate_embeddings(embeddings, labels_enc, list(le.classes_))

    # Save
    emb_path = out_dir / "clinical_embeddings.npy"
    np.save(str(emb_path), embeddings)
    print(f"\nSaved: {emb_path}  shape={embeddings.shape}")

    feat_cols = [CONFIG["patient_id_col"], CONFIG["label_col"]] + flag_cols
    df[feat_cols].to_csv(out_dir / "clinical_features.csv", index=False)
    print(f"Saved: clinical_features.csv")

    df[[CONFIG["patient_id_col"], CONFIG["label_col"], "full_report"]].to_csv(
        out_dir / "full_reports.csv", index=False
    )
    print(f"Saved: full_reports.csv")

    print(f"\nDone — all outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
