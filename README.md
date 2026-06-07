# Multimodal AI for Oncology

A multimodal deep learning pipeline for melanocytic skin lesion classification combining **dermoscopy image features** (PanDerm ViT) and **clinical pathology text** (BioClinicalBERT) via late fusion.

---

## Results

| Model | Balanced Accuracy | Macro AUC | Macro F1 |
|-------|:-----------------:|:---------:|:--------:|
| PanDerm pre-trained (no fine-tuning) | 0.524 ± 0.066 | 0.723 ± 0.055 | 0.506 ± 0.065 |
| PanDerm fine-tuned (image only) | 0.546 ± 0.040 | 0.699 ± 0.041 | 0.523 ± 0.047 |
| **PanDerm + BioClinicalBERT (fusion)** | **0.603 ± 0.060** | **0.778 ± 0.057** | **0.576 ± 0.061** |

> Fusion evaluated on 138/177 patients with both dermoscopy images and clinical reports.  
> All results: patient-level, 5-fold cross-validation, logistic regression classifier.

**Per-class results (fusion model):**

| Class | F1 | AUC |
|-------|:--:|:---:|
| Dysplastic Nevus | 0.543 ± 0.131 | 0.720 ± 0.084 |
| Melanoma Stage IA | 0.690 ± 0.079 | 0.814 ± 0.108 |
| Melanoma In Situ | 0.337 ± 0.100 | 0.564 ± 0.061 |

---

## Overview

| Modality | Model | Output |
|----------|-------|--------|
| Dermoscopy images | PanDerm Large ViT (fine-tuned) | (N, 1024) patient-level features |
| Pathology reports | BioClinicalBERT | (N, 768) text embeddings |
| Fusion | Logistic Regression | 3-class prediction |

**Classes:**
- Dysplastic Nevus (DN)
- Melanoma In Situ (Minsitu)
- Melanoma Stage IA (MIA)

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
├── clinical/
│   └── clinical_pipeline.py              # BioClinicalBERT embedding pipeline
│
├── panderm/
│   ├── panderm-finetuning-colab.ipynb    # Fine-tuning notebook (Google Colab)
│   ├── panderm_finetuning.py             # Fine-tuning script
│   ├── extract_and_evaluate.py           # Feature extraction + k-fold evaluation
│   ├── prepare_data.py                   # Dataset manifest + fold splits
│   ├── make_panderm_csv.py               # Create PanDerm-format CSVs
│   ├── segment_lesions.py                # Lesion segmentation (preprocessing)
│   └── config.py                         # Shared constants
│
├── fusion/
│   ├── late_fusion_colab.ipynb           # Fusion notebook (Google Colab)
│   └── late_fusion.py                    # Fusion script
│
├── clinical-biobert-pipeline.ipynb       # Clinical pipeline (Kaggle notebook)
└── README.md
```

---

## GPU Requirements

| Step | Script | Recommended GPU | Approximate Time |
|------|--------|----------------|-----------------|
| PanDerm fine-tuning (per fold) | `panderm_finetuning.py` | NVIDIA T4 (16 GB) or better | ~30–60 min/fold |
| PanDerm fine-tuning (5 folds total) | `panderm_finetuning.py` | NVIDIA T4 (16 GB) or better | ~3–5 hours |
| Feature extraction (per fold) | `extract_and_evaluate.py` | NVIDIA T4 (16 GB) | ~5 min/fold |
| BioClinicalBERT embeddings | `clinical_pipeline.py` | CPU or any GPU | ~2–5 min |
| Late fusion evaluation | `late_fusion.py` | CPU (no GPU needed) | < 1 min |

**Free-tier options:**
- **Google Colab T4** — used for this project. ~3–4 hours compute per session, resets after ~24 hours.
- **Kaggle T4** — used for clinical pipeline. 30 GPU hours/week free.

> Note: Feature extraction caches results to disk, so GPU is only needed once per fold.

---

## 1. Clinical Text Pipeline

Extracts dense semantic embeddings from pathology reports using **BioClinicalBERT** (`emilyalsentzer/Bio_ClinicalBERT`), trained on MIMIC-III clinical notes.

**Steps:**
1. Load pathology reports (diagnosis + macroscopic description)
2. Encode with BioClinicalBERT (mean pooling over tokens, max 512 tokens)
3. Validate with UMAP, cosine similarity, k-NN LOO accuracy
4. Save `clinical_embeddings.npy` (177, 768)

**Run:**
```bash
python clinical/clinical_pipeline.py
```

**Requirements:**
```bash
pip install transformers torch openpyxl umap-learn scikit-learn tqdm
```

---

## 2. PanDerm Fine-Tuning

Fine-tunes **PanDerm Large ViT** ([SiyuanYan1/PanDerm](https://github.com/SiyuanYan1/PanDerm)) on dermoscopy images using 5-fold cross-validation with patient-level stratified splits.

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| Model | PanDerm Large (ViT-L/16) |
| Feature dimension | 1024 |
| Pretrained checkpoint | panderm_ll_data6_checkpoint-499.pth |
| Epochs | 50 |
| Warmup epochs | 5 |
| Batch size | 32 |
| Layer decay | 0.65 |
| Drop path | 0.2 |
| Weight decay | 0.05 |
| Mixup / CutMix | 0.8 / 1.0 |
| Optimizer | AdamW |
| Classes | 3 |

**Run on Google Colab (recommended):**
1. Clone PanDerm repo: `git clone https://github.com/SiyuanYan1/PanDerm.git`
2. Open `panderm/panderm-finetuning-colab.ipynb`
3. Mount Google Drive and add `WANDB_API_KEY` to Colab Secrets
4. Edit `DRIVE_ROOT` in Cell 5 to match your Drive layout
5. Run cells sequentially: 1 → 2 → restart → 3 → 4 → 5 → 6 → 6b → 7 → 8 → 9

**Run as script:**
```bash
# Edit CONFIG paths at top of file first
python panderm/panderm_finetuning.py
```

**Requirements:**
```bash
pip install timm==0.9.16 "numpy<2.0" wandb open_clip_torch
pip install torch torchvision scikit-learn umap-learn
```

---

## 3. Feature Extraction + Evaluation

Loads each fold's fine-tuned checkpoint, extracts CLS token features, aggregates to patient level, and evaluates with logistic regression.

**Run:**
```bash
# Edit CONFIG paths at top of file first
python panderm/extract_and_evaluate.py
```

**Key design decisions:**
- Each fold uses its **own fine-tuned checkpoint** — no data leakage (test images never seen during that fold's training)
- Uses **segmented lesion images** for feature extraction (consistent with baseline)
- **Mean pooling** across images per patient group → one 1024-dim vector per patient
- 3 patients appear in two classes → 166 patient groups from 163 unique patients

**Outputs:**
```
results/features/
    fold{i}_image_features.npy        (562, 1024)
    patient_features_fold{i}.npy      (166, 1024)
    patient_labels.npy
    patient_group_ids.npy
    group_fold_mapping.csv
results/
    kfold_results_finetuned.csv
    fold_{i}_report.txt
    confusion_matrix_aggregate.png
    roc_curves_mean.png
    fold_accuracy_bars.png
```

---

## 4. Late Fusion

Concatenates PanDerm image features and BioClinicalBERT text embeddings and evaluates with logistic regression.

**Run:**
```bash
# Edit CONFIG paths at top of file first
python fusion/late_fusion.py
```

**How it works:**
```python
# For each fold:
img_feats  = patient_features_fold{i}.npy   # (138, 1024)
text_feats = clinical_embeddings.npy         # (138,  768)
fused      = concatenate([img_feats, text_feats], axis=1)  # (138, 1792)
# Train logistic regression on train split, evaluate on test split
```

**Outputs:**
```
fusion_results/
    fusion_kfold_results.csv
    fusion_confusion_matrix.png
    fusion_roc_curves.png
    fusion_umap.png
```

---

## Data

- **177** melanocytic skin lesion cases
- **3 classes:** DN, MIA, Melanoma In Situ
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
