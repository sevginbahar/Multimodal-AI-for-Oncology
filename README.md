# Multimodal AI for Oncology

A multimodal deep learning pipeline for melanocytic skin lesion classification combining **dermoscopy image features** (PanDerm ViT) and **clinical pathology text** (BioClinicalBERT) via late fusion.

---

## Results

All results: patient-level, 5-fold cross-validation, logistic regression classifier.

| Model | Balanced Accuracy | Macro AUC | Macro F1 |
|-------|:-----------------:|:---------:|:--------:|
| PanDerm (frozen) | 0.524 ± 0.066 | 0.723 ± 0.055 | 0.506 ± 0.065 |
| PanDerm (fine-tune) | 0.546 ± 0.040 | 0.699 ± 0.041 | 0.523 ± 0.047 |
| BioClinicalBERT (pathology reports) | 0.897 ± 0.015 | 0.976 ± 0.015 | 0.898 ± 0.016 |
| Late Fusion (fine-tune + pathology) | 0.603 ± 0.060 | 0.778 ± 0.057 | 0.576 ± 0.061 |
| **Late Fusion (frozen + pathology)** | **0.747 ± 0.082** | **0.903 ± 0.026** | **0.733 ± 0.084** |

> BioClinicalBERT and fusion evaluated on 177 and 138 patients respectively (those with both modalities).
> Paired t-tests confirm Late Fusion (frozen) significantly outperforms PanDerm (frozen) on all metrics (p=0.002–0.005).
> BioClinicalBERT alone significantly outperforms the best fusion model on all metrics (p=0.001–0.017).

**Significance tests (paired t-test, 5-fold, Macro AUC):**

| Comparison | Δ | 95% CI | p |
|------------|:-:|:------:|:-:|
| Late Fusion (fine-tune) vs PanDerm (fine-tune) | +0.079 | [+0.018, +0.141] | 0.023 * |
| Late Fusion (frozen) vs PanDerm (frozen) | +0.180 | [+0.109, +0.250] | 0.002 * |
| Late Fusion (frozen) vs Late Fusion (fine-tune) | +0.124 | [+0.053, +0.196] | 0.009 * |
| BioClinicalBERT vs Late Fusion (frozen) | +0.074 | [+0.049, +0.098] | 0.001 * |
| BioClinicalBERT vs PanDerm (fine-tune) | +0.277 | [+0.230, +0.324] | <0.001 * |

\* p < 0.05

DN = Dysplastic Nevus · MIA = Melanoma Stage IA · Minsitu = Melanoma In Situ

---

## Overview

| Modality | Model | Output |
|----------|-------|--------|
| Dermoscopy images | PanDerm Large ViT (fine-tuned) | (N, 1024) patient-level features |
| Pathology reports | BioClinicalBERT | (N, 768) text embeddings |
| Fusion | Logistic Regression | 3-class prediction |

**Classes:** Dysplastic Nevus (DN) · Melanoma In Situ (Minsitu) · Melanoma Stage IA (MIA)

**Dataset:** 177 melanocytic skin lesion cases with paired dermoscopy images and pathology reports.

---

## Pipeline

```
Dermoscopy Images                    Pathology Reports
       │                                     │
       ▼                                     ▼
PanDerm Large ViT                  BioClinicalBERT
(fine-tuned, 5-fold CV)            (mean pooling)
       │                                     │
       ▼                                     ▼
Image Features (1024-dim)      Text Embeddings (768-dim)
       │                                     │
       └──────────────┬──────────────────────┘
                      ▼
               Late Fusion
          (concatenate → 1792-dim)
                      │
                      ▼
            Logistic Regression
                      │
                      ▼
            3-Class Prediction
       (DN / Minsitu / MIA)
```

---

## Repository Structure

```
Multimodal-AI-for-Oncology/
├── config.py                        # All paths and hyperparameters — edit this first
├── run_pipeline.py                  # End-to-end pipeline orchestrator
│
├── panderm/
│   ├── prepare_data.py              # Scan dataset, create manifest + k-fold splits
│   ├── segment_lesions.py           # Lesion segmentation (LAB + Otsu + morphological closing)
│   ├── make_panderm_csv.py          # Convert manifest to PanDerm-format fold CSVs
│   ├── panderm_finetuning.py        # Fine-tune PanDerm (5-fold, calls PanDerm repo)
│   ├── extract_features.py          # Extract CLS token features — requires GPU
│   └── evaluate.py                  # Logistic regression + plots + attention maps — CPU only
│
├── clinical/
│   └── clinical_pipeline.py         # BioClinicalBERT embeddings from pathology reports
│
└── fusion/
    └── late_fusion.py               # Late fusion: image + text → logistic regression
```

---

## Quickstart

### 1. Setup

```bash
# Clone this repo
git clone <this-repo>
cd Multimodal-AI-for-Oncology

# Clone PanDerm
git clone https://github.com/SiyuanYan1/PanDerm.git

# Install dependencies
pip install timm==0.9.16 "numpy<2.0" torch torchvision
pip install transformers scikit-learn umap-learn wandb
pip install opencv-python matplotlib seaborn pandas tqdm
```

### 2. Configure paths

Edit **`config.py`** — this is the only file you need to change:

```python
DATA_ROOT        = Path("/path/to/dermoscopy")           # raw images, one folder per class
PANDERM_REPO     = Path("/path/to/PanDerm")              # cloned PanDerm repo
CHECKPOINT_LARGE = Path("/path/to/panderm_ll_data6_checkpoint-499.pth")
PIPELINE_DIR     = Path("/path/to/improved_pipeline")    # outputs go here
```

### 3. Run the pipeline

```bash
# Full pipeline (with fine-tuning)
python run_pipeline.py --stage all

# Full pipeline (pretrained features only — no fine-tuning)
python run_pipeline.py --stage all --no-finetune

# Resume from a specific stage
python run_pipeline.py --stage all --skip-existing

# Individual stages
python run_pipeline.py --stage prepare
python run_pipeline.py --stage segment
python run_pipeline.py --stage make_csv
python run_pipeline.py --stage finetune
python run_pipeline.py --stage extract_features   # GPU job
python run_pipeline.py --stage evaluate           # CPU job
python run_pipeline.py --stage clinical_modality
python run_pipeline.py --stage fusion
```

---

## Stage Details

### 1. `prepare_data.py`
Scans the dermoscopy dataset, creates `dataset_manifest.csv` with patient-level stratified 5-fold splits (StratifiedGroupKFold, no patient leakage across folds).

**Output:** `results/dataset_manifest.csv`

### 2. `segment_lesions.py`
Segments lesions from the background using LAB colour space + Otsu thresholding + morphological closing. Crops to the lesion bounding box with a 10% margin. Falls back to the full image if the mask is too small.

**Output:** `segmented_cache/` + updated `segmented_path` column in manifest

### 3. `make_panderm_csv.py`
Converts the manifest into 5 fold CSVs with `image`, `label`, `split` columns in the format expected by PanDerm's `run_class_finetuning.py`.

**Output:** `cross-fold-csv/panderm_finetuning_fold{0-4}.csv`

### 4. `panderm_finetuning.py`
Fine-tunes PanDerm Large (ViT-L/16) on the segmented images for each fold.

| Parameter | Value |
|-----------|-------|
| Model | PanDerm Large (ViT-L/16) |
| Epochs | 50 (warmup: 5) |
| Batch size | 32 |
| Layer decay | 0.65 |
| Drop path | 0.2 |
| Mixup / CutMix | 0.8 / 1.0 |
| Optimizer | AdamW |

**Output:** `results/results_fold{0-4}/checkpoint-best.pth`

### 5. `extract_features.py` — GPU required
Loads each fold's fine-tuned checkpoint, extracts 1024-dim CLS token features, and aggregates to patient level by mean pooling across images.

**Output:** `features/fold{i}_image_features.npy`, `features/patient_features_fold{i}.npy`

### 6. `evaluate.py` — CPU only
Reads saved features, runs logistic regression k-fold evaluation, and generates all plots.

**Outputs:**
```
results/
    kfold_results_finetuned.csv
    fold_{i}_report.txt
    confusion_matrix_aggregate.png
    roc_curves_mean.png
    fold_accuracy_bars.png
    umap_patient_features.png
    attention_map_examples_finetuned.png
```

### 7. `clinical_pipeline.py`
Encodes pathology reports with BioClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`), mean-pooled over tokens.

**Output:** `clinical_outputs/clinical_embeddings.npy` (N, 768)

### 8. `late_fusion.py`
Concatenates image features (1024-dim) and text embeddings (768-dim) → 1792-dim, then evaluates with logistic regression. Supports both frozen and fine-tuned image features via `--mode`.

```bash
python fusion/late_fusion.py --mode finetune   # fine-tuned PanDerm + text
python fusion/late_fusion.py --mode frozen     # frozen PanDerm + text
```

**Output:** `fusion_results/fusion_kfold_results.csv`, fusion plots

---

## GPU Requirements

| Stage | Script | GPU | Approximate Time |
|-------|--------|-----|-----------------|
| Fine-tuning (per fold) | `panderm_finetuning.py` | Required (≥16 GB) | ~30–60 min/fold |
| Fine-tuning (5 folds) | `panderm_finetuning.py` | Required (≥16 GB) | ~3–5 hours |
| Feature extraction | `extract_features.py` | Required | ~5 min/fold |
| Evaluation + plots | `evaluate.py` | Not needed | < 1 min |
| Clinical embeddings | `clinical_pipeline.py` | Optional | ~2–5 min |
| Late fusion | `late_fusion.py` | Not needed | < 1 min |

---

## Data

- **177** melanocytic skin lesion cases
- **3 classes:** Dysplastic Nevus (DN), Melanoma Stage IA (MIA), Melanoma In Situ (Minsitu)
- Paired dermoscopy images + pathology reports
- Reports in English, Spanish, and Catalan (translated to English)
- 5-fold cross-validation splits (patient-level, no data leakage)
- 138/177 patients have both modalities (used for fusion)

---

## Requirements

| Library | Version |
|---------|---------|
| Python | 3.10+ |
| PyTorch | 2.x |
| timm | **0.9.16** (required for PanDerm) |
| transformers | 4.x |
| scikit-learn | 1.x |
| umap-learn | 0.5+ |
| wandb | latest |
| numpy | < 2.0 |
| opencv-python | 4.x |

---

## Citation

If you use PanDerm, please cite:

```bibtex
@article{panderm2024,
  title={PanDerm: A Foundation Model for Dermatology},
  author={Yan, Siyuan et al.},
  journal={Nature Medicine},
  year={2024}
}
```

---

## Author

**Bahar Sevgin**
Queen Mary University of London
Research Assistant — Maiques Lab
