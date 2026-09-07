# Multimodal AI for Oncology

A multimodal deep learning pipeline for melanocytic skin lesion classification combining **dermoscopy image features** (PanDerm ViT) and **clinical pathology text** (BioClinicalBERT) via late fusion.

Three roughly balanced diagnostic categories:

| Key | Class | |
|-----|-------|--|
| `DN` | Dysplastic Nevus | benign |
| `Minsitu` | Melanoma In Situ (MIS) | pre-invasive |
| `MIA` | Melanoma Stage IA | early invasive |

---

## Dataset

Lesion-linked melanocytic lesion cohort (Maiques Lab).

| | Count |
|---|---|
| Dermoscopy images | **547** (482 `.jpg` + 65 `.tif`) |
| Patients with images | **163** — DN 56 · MIA 57 · Minsitu 50 |
| Pathology reports (one per patient) | **174** |
| Patients with **both** modalities (fusion set) | **156** — DN 56 · MIA 50 · Minsitu 50 |
| Median images / patient | 3 |

- Reports arrive as harmonised CSV columns (diagnostic summary + macroscopic description), originally English / Spanish / Catalan, translated to English.
- 5-fold cross-validation, **patient-level** (`StratifiedGroupKFold`) — no patient appears in more than one fold; the 3 multi-diagnosis patients are kept whole.
- 11 patient folders in the raw export contain no image and are excluded.

---

## Results

Current stage: **frozen encoders + logistic-regression head**, patient-level 5-fold CV.
Fine-tuning is implemented (`--stage finetune`) but not part of these results.

| Model | Eval set | Balanced Acc | Macro AUC | Macro F1 | Cohen κ |
|-------|:--------:|:------------:|:---------:|:--------:|:-------:|
| PanDerm Large (frozen) — images | 163 pt | 0.569 ± 0.061 | 0.729 ± 0.033 | 0.561 ± 0.058 | 0.361 ± 0.091 |
| **BioClinicalBERT — reports** | 174 pt | **0.852 ± 0.062** | **0.953 ± 0.023** | **0.852 ± 0.060** | **0.776 ± 0.091** |
| Late Fusion (frozen PanDerm + BioClinicalBERT) | 156 pt | 0.739 ± 0.060 | 0.896 ± 0.036 | 0.738 ± 0.059 | 0.612 ± 0.089 |

± = standard deviation across the 5 folds.

### Per-class (F1 / AUC, mean ± SD)

| Class | PanDerm frozen | BioClinicalBERT | Late Fusion (frozen) |
|-------|:--------------:|:---------------:|:--------------------:|
| Dysplastic Nevus | 0.556 ± 0.090 / 0.784 ± 0.057 | 0.897 ± 0.037 / 0.974 ± 0.016 | 0.767 ± 0.069 / 0.941 ± 0.019 |
| Melanoma Stage IA | 0.687 ± 0.090 / 0.813 ± 0.078 | 0.858 ± 0.081 / 0.970 ± 0.037 | 0.813 ± 0.082 / 0.933 ± 0.030 |
| Melanoma In Situ | 0.441 ± 0.140 / 0.588 ± 0.082 | 0.800 ± 0.082 / 0.914 ± 0.035 | 0.635 ± 0.087 / 0.815 ± 0.075 |

### Significance

| Comparison | Metric | Δ | 95% CI | p |
|------------|:------:|:-:|:------:|:-:|
| Late Fusion (frozen) vs PanDerm (frozen) | Macro AUC | +0.168 | [+0.097, +0.238] | 0.003 * |
| Late Fusion (frozen) vs PanDerm (frozen) | Balanced Acc | +0.169 | [+0.100, +0.239] | 0.003 * |

Paired t-test across the 5 shared patient-level folds. \* p < 0.05.

### Backbone benchmark — frozen encoders (`baselines/benchmark_backbones.py`)

Same segmented images, same folds, same logreg head — only the frozen encoder changes.

| Encoder | Pretraining | Macro AUC | Balanced Acc | vs PanDerm-frozen (Macro AUC) |
|---------|-------------|:---------:|:------------:|:-----------------------------:|
| PanDerm Large ViT-L/16 | dermatology SSL (~2 M images) | 0.729 ± 0.033 | 0.569 ± 0.061 | — |
| ConvNeXt-Tiny | ImageNet-21k | 0.731 ± 0.026 | 0.563 ± 0.031 | Δ +0.002, p = 0.91 |
| ConvNeXt-Small | ImageNet-21k | 0.729 ± 0.052 | 0.521 ± 0.088 | Δ +0.000, p = 0.99 |

Paired t-test, 5 shared folds. No comparison is significant on any metric.

- **The "ConvNeXt beats ViT on small data" claim does not transfer here** — ConvNeXt matches PanDerm, it does not beat it.
- **PanDerm's dermatology pretraining gives no measurable advantage over generic ImageNet features** at the frozen linear-probe level. All three encoders are pinned at Macro AUC ≈ 0.73.
- That ≈ 0.73 ceiling is the **frozen-features / small-data regime**, not the choice of encoder. Swapping the frozen backbone is not a lever for the image modality; ConvNeXt-Small (bigger, noisier, no gain) shows the capacity penalty at n = 547.

### Reading the results

- **Text ≫ images.** The pathology reports are near-linearly separable (Macro AUC 0.95); the frozen dermoscopy features sit far below (0.73).
- **Fusion helps the image side but is dragged down by it.** Adding images to text *lowers* every metric vs text alone (0.896 vs 0.953 AUC) — the current fusion (per-modality PCA-8 → concat-16 → logreg) gives the weak modality too much weight. Fusion still beats images-alone decisively (p = 0.003).
- **Melanoma In Situ is the bottleneck on every modality.** It is the hardest class for images (AUC 0.59, ≈ chance), text (0.91), and fusion (0.82) — the DN ↔ MIS boundary is where the signal is thin.
- A strict paired test of text vs fusion is not available: the text CV runs on all 174 report-patients with a label-stratified split, not the 156-patient image folds.

---

## Overview

| Modality | Model | Output |
|----------|-------|--------|
| Dermoscopy images | PanDerm Large ViT-L/16 (frozen, pretrained checkpoint) | (N, 1024) patient-level features |
| Pathology reports | BioClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`) | (N, 768) mean-pooled embeddings |
| Fusion | per-modality StandardScaler → PCA(8) → concat(16) → Logistic Regression | 3-class prediction |

---

## Pipeline

```
Dermoscopy Images                    Pathology Reports
       │                                     │
       ▼                                     ▼
 Lesion segmentation                report_cleaning.py
 (LAB + Otsu + close)               (termfilt: strip dx/stage terms)
       │                                     │
       ▼                                     ▼
PanDerm Large ViT-L/16              BioClinicalBERT
(frozen; fine-tune optional)        (mean pooling)
       │                                     │
       ▼                                     ▼
Image features (1024-dim)      Text embeddings (768-dim)
  mean-pool over patient's images        │
       │                                 │
       ├───────── evaluate.py            ├───────── clinical_pipeline.py
       │          (image-only LR)        │          (text-only LR)
       │                                 │
       └──────────────┬──────────────────┘
                      ▼
              late_fusion.py
       per-modality PCA(8) → concat(16)
                      │
                      ▼
             Logistic Regression
                      │
                      ▼
             3-class prediction
          (DN / Minsitu / MIA)

  baselines/benchmark_backbones.py — swaps the frozen encoder
  (ConvNeXt-T/S by default) keeping folds + head fixed
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
│   ├── make_panderm_csv.py          # Convert manifest to PanDerm-format fold CSVs (fine-tune only)
│   ├── panderm_finetuning.py        # Fine-tune PanDerm (5-fold, calls PanDerm repo)
│   ├── extract_features.py          # Extract features (frozen or fine-tuned) — GPU
│   └── evaluate.py                  # Logistic regression + plots + attention maps — CPU
│
├── clinical/
│   ├── clinical_pipeline.py         # BioClinicalBERT embeddings from pathology reports
│   ├── report_cleaning.py           # Graded leakage control (orig → termfilt → diagdrop → fact → notext)
│   ├── ocr_reports.py               # OCR front-end for scanned PDFs (TCGA) — optional
│   └── tcga_extract.py              # Pull text layer from TCGA report PDFs — optional
│
├── fusion/
│   └── late_fusion.py               # Late fusion: image + text → logistic regression
│
└── baselines/
    ├── benchmark_backbones.py       # Frozen ConvNeXt-T/S vs PanDerm (same folds + head; ViT/DINOv2 optional)
    └── README.md                    # Benchmark rationale + how to read it
```

---

## Quickstart

### 1. Environment (Python 3.10 or 3.11 — **not 3.12+**)

`numpy<2.0`, `torch`, and `timm==0.9.16` (fine-tune stage) have no wheels for newer Python.

```bash
conda create -n oncology python=3.11 -y
conda activate oncology

git clone https://github.com/SiyuanYan1/PanDerm.git   # anywhere; point config.py at it

pip install "numpy<2.0"
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install pandas scikit-learn scipy scikit-image opencv-python Pillow matplotlib seaborn tqdm umap-learn
pip install transformers==4.38.2 open_clip_torch timm

# fine-tune stage only — pins timm back to 0.9.16
pip install -r <PanDerm>/classification/requirements.txt
```

> The frozen pipeline and the backbone benchmark run on any recent `timm` (1.0.x is fine).
> `timm==0.9.16` is required **only** for `--stage finetune`.

### 2. Configure paths

Edit **`config.py`** — the only file you need to change. Example (Windows):

```python
DATA_ROOT        = Path(r"C:\...\phd_data\dermoscopy\dermoscopy")   # DATA_ROOT/{DN,MIA,Minsitu}/{patient_id}/*.jpg
PANDERM_REPO     = Path(r"C:\...\PanDerm")
CHECKPOINT_LARGE = Path(r"C:\...\panderm_ll_data6_checkpoint-499.pth")
OUTPUT_DIR       = Path(r"C:\...\phd_data\dermoscopy_outputs")
CLINICAL_INPUT   = Path(r"C:\...\phd_data\harmonized_pathology_reports.csv")
CLINICAL_DIR     = Path(r"C:\...\phd_data\clinical_outputs")
```

### 3. Run — frozen pipeline (no GPU fine-tuning)

```bash
python run_pipeline.py --stage prepare
python run_pipeline.py --stage segment
python run_pipeline.py --stage extract_features --no-finetune
python run_pipeline.py --stage evaluate         --no-finetune
python run_pipeline.py --stage clinical_modality
python run_pipeline.py --stage fusion           --no-finetune
python baselines/benchmark_backbones.py
```

`--no-finetune` forces `--mode frozen` on `extract_features`, `evaluate`, and `fusion` when they are run individually. `make_csv` and `finetune` are skipped.

### 3b. Full pipeline with fine-tuning

```bash
pip install timm==0.9.16       # required
python run_pipeline.py --stage all              # prepare → … → finetune → … → fusion
python run_pipeline.py --stage all --skip-existing
```

---

## Stage Details

### 1. `prepare_data.py`
Scans `DATA_ROOT/{class}/{patient_id}/`, builds `dataset_manifest.csv` with patient-level stratified 5-fold splits (`StratifiedGroupKFold`, groups = `patient_id`). Validates no leakage.

**Output:** `OUTPUT_DIR/dataset_manifest.csv`

### 2. `segment_lesions.py`
LAB colour space + Otsu + morphological closing; crops to the lesion bounding box + 10% margin; falls back to the full image when the mask is < 1% of the frame.

**Output:** `SEGMENTED_DIR/` + `segmented_path` column in the manifest + `segmentation_qc_montage.png`

### 3. `make_panderm_csv.py` *(fine-tune only)*
Converts the manifest to 5 fold CSVs (`image`, `label`, `split`) for PanDerm's `run_class_finetuning.py`.

**Output:** `CSV_DIR/panderm_finetuning_fold{0-4}.csv`

### 4. `panderm_finetuning.py` — GPU
Fine-tunes PanDerm Large (ViT-L/16) per fold. Epochs 50 (warmup 5), batch 32, layer decay 0.65, drop-path 0.2, Mixup/CutMix 0.8/1.0, AdamW.

**Output:** `OUTPUT_DIR/results_fold{0-4}/checkpoint-best.pth`

> ⚠️ Needs ~16 GB VRAM at batch 32. On smaller GPUs set `BATCH_SIZE = 8` and add `--update_freq 4 --amp` in `panderm_finetuning.py`, or use the Base checkpoint (`MODEL_VARIANT = "base"`). Shells out to `sed`/`grep` — run from Git Bash on Windows or apply the `weights_only=False` patch to `run_class_finetuning.py` by hand.

### 5. `extract_features.py` — GPU
`--mode frozen`: one pass with the pretrained checkpoint, reused across folds.
`--mode finetune`: per-fold checkpoint.
Extracts 1024-dim features, mean-pools to patient level.

**Output:** `FEATURES_DIR/frozen_image_features.npy`, `patient_features_fold{i}.npy`, `patient_labels.npy`, `patient_group_ids.npy`, `group_fold_mapping.csv`

Loading the self-supervised checkpoint prints `missing=2, unexpected=92` — expected: the 92 are the MAE pretraining decoder (discarded), the 2 are the unused classification head.

### 6. `evaluate.py` — CPU
Logistic regression (nested-CV `C ∈ {0.01, 0.03, 0.1, 0.3, 1.0}`, `class_weight="balanced"`) on the patient-level folds; all plots.

**Output:** `OUTPUT_DIR/kfold_results_{frozen,finetuned}.csv`, `fold_{i}_report.txt`, `confusion_matrix_aggregate.png`, `roc_curves_mean.png`, `fold_accuracy_bars.png`, `umap_patient_features.png`, (fine-tune only) `attention_map_examples_*.png`

> On the frozen path the run ends with `Attention maps failed: No checkpoint found` — expected, there are no per-fold checkpoints. Metrics and all other plots are produced.

### 7. `clinical_pipeline.py`
BioClinicalBERT, mean-pooled over tokens, on reports cleaned at the `termfilt` level (regex-strips diagnosis / staging / invasion terms). Includes k-NN LOO + permutation test + length-confound check + its own 5-fold LR.

**Output:** `CLINICAL_DIR/clinical_embeddings.npy` (174, 768), `full_reports.csv`, `text_only_kfold_results.csv`, `umap_embeddings.png`

### 8. `late_fusion.py`
Per-modality `StandardScaler → PCA(8)` (fit on train only) → concat(16) → logistic regression, on the 156 patients with both modalities. `--mode {frozen,finetune}`.

**Output:** `OUTPUT_DIR/fusion_results/fusion_kfold_results.csv` + confusion / ROC / UMAP plots

### 9. `baselines/benchmark_backbones.py`
Swaps the frozen encoder — **ConvNeXt-Tiny/Small** by default (ImageNet-21k) — keeping the folds, patient pooling, and logreg head identical to stage 6. Tests whether architecture or dermatology pretraining is what matters at this sample size. Pulls in `kfold_results_frozen.csv` so PanDerm sits in the same table with paired t-tests. `vit_b16` and `dinov2_b` are selectable via `--backbones` but off by default (a generic ViT is expected to lose to a domain-pretrained one). See `baselines/README.md`.

**Output:** `baselines/results/backbone_benchmark_perfold.csv`, `backbone_benchmark_summary.csv`, `significance_vs_panderm.csv`, `backbone_comparison_seg.png`

**Finding:** ConvNeXt-T/S and PanDerm-frozen are statistically indistinguishable (Macro AUC ≈ 0.73, all p > 0.9). See Results → Backbone benchmark.

---

## Compute

Reference run: RTX 3060 Laptop (6 GB), Ryzen 5 5600H, 16 GB RAM, Windows 11.

| Stage | GPU | Time (this machine) |
|-------|-----|---------------------|
| prepare | — | ~1.5 min |
| segment | — | ~11 min (CPU, single-thread) |
| extract_features (frozen) | yes | ~1.5 min |
| evaluate | — | ~1.5 min |
| clinical_modality | yes (first run downloads ~440 MB) | ~4 min |
| fusion | — | < 1 min |
| benchmark_backbones (ConvNeXt-T/S) | yes (downloads ~300 MB of weights) | ~2 min |
| **fine-tuning (5 folds)** | **≥ 16 GB ideal** | not run here |

Optional ViT baselines: `python baselines/benchmark_backbones.py --backbones dinov2_b --batch-size 8` (runs at 518 px; the `--batch-size 8` avoids OOM on 6 GB — the ConvNeXt features stay cached).

---

## Requirements

| Library | Version |
|---------|---------|
| Python | **3.10 or 3.11** |
| numpy | **< 2.0** |
| PyTorch | 2.2.2 (cu121) |
| torchvision | 0.17.2 |
| timm | 1.0.x for frozen pipeline · **0.9.16** for `--stage finetune` |
| transformers | 4.38.2 |
| open_clip_torch | any (imported by PanDerm's `models` package) |
| scikit-learn / scipy / scikit-image | 1.x |
| umap-learn | 0.5+ |
| opencv-python | 4.x |
| matplotlib / seaborn / pandas / tqdm / Pillow | latest |
| pymupdf, pytesseract | optional — `clinical/ocr_reports.py` only |

---

## Citation

```bibtex
@article{panderm2024,
  title={PanDerm: A Foundation Model for Dermatology},
  author={Yan, Siyuan and others},
  journal={Nature Medicine},
  year={2024}
}
```

---

## Author

**Bahar Sevgin**
Queen Mary University of London — Research Assistant, Maiques Lab
