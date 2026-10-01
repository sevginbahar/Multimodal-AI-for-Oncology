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
- 5-fold cross-validation, **patient-level** (`StratifiedGroupKFold`, seed 42) — no patient appears in more than one fold. `prepare_data.py` is deterministic: re-running it reproduces the same folds.
- 11 patient folders in the raw export contain no image and are excluded.

> **Known data issue (to be fixed before final results):** two DN patient folders contain byte-identical copies of another patient's images — `14165882` = images of `14149067`, `14173497` = images of `14293082` — and these pairs fall in different folds (train/test leakage affecting all methods equally). Two further within-patient duplicates exist (`10136423 (7) - Copy.JPG`, two identical files in `Minsitu/14252811`). All results below still include them.

---

## Results

All numbers: patient-level 5-fold CV, mean ± SD across folds. With ~32 test patients per fold, differences below ≈ 0.05 are within noise; none of the image-model comparisons below has been tested formally yet.

### Image model (frozen PanDerm-L unless stated, Sep 2026 run)

| Experiment | Segmentation | Balanced Acc | Macro AUC | Macro F1 |
|---|---|:-:|:-:|:-:|
| Frozen – CLS token | none (raw image) | 0.570 ± 0.087 | 0.751 ± 0.055 | 0.568 ± 0.085 |
| Frozen – mean of all patch tokens | none (raw image) | 0.577 ± 0.076 | 0.753 ± 0.053 | 0.573 ± 0.075 |
| Frozen – CLS on crop | Otsu v1 | 0.563 ± 0.058 | 0.729 ± 0.033 | 0.556 ± 0.058 |
| Frozen – CLS on crop | Otsu v2 | 0.485 ± 0.083 | 0.699 ± 0.067 | 0.479 ± 0.074 |
| Frozen – CLS on crop | SAM2 v1 | 0.504 ± 0.061 | 0.695 ± 0.054 | 0.492 ± 0.059 |
| Frozen – CLS on crop | SAM2 v1 + preprocessing | 0.527 ± 0.059 | 0.697 ± 0.016 | 0.521 ± 0.074 |
| Frozen – CLS on crop | SAM2 v2 | 0.521 ± 0.040 | 0.732 ± 0.049 | 0.519 ± 0.035 |
| Mask-guided pooling | Otsu v2 masks | 0.589 ± 0.035 | 0.768 ± 0.041 | 0.588 ± 0.036 |
| Mask-guided pooling | SAM2 v1 + preprocessing masks | 0.534 ± 0.037 | 0.729 ± 0.034 | 0.531 ± 0.033 |
| **Mask-guided pooling** | **SAM2 v2 masks** | **0.609 ± 0.029** | **0.767 ± 0.047** | **0.607 ± 0.029** |
| Partial fine-tune (blocks 18–23 + head) | none (raw image) | 0.528 ± 0.058 | 0.713 ± 0.047 | 0.518 ± 0.062 |

Reading the image results:

- **Cropping to the segmented lesion hurts**, even with better masks: it removes lesion size, the lesion–skin border and surrounding skin, squashes the lesion box to a square and adds a black background PanDerm was not pretrained on.
- **Segmentation helps as guidance, not as a crop.** Mask-guided pooling feeds the raw image to PanDerm and uses the mask only to weight its patch tokens. The control (plain mean of all patch tokens, 0.577) shows the gain comes from the mask, not from pooling itself. Bad masks hurt: SAM2 v1 + preprocessing masks score below no mask at all.
- **Partial fine-tuning does not beat frozen features** at n = 547: validation loss is lowest at epochs 3–8, then the model overfits (75.6 M trainable parameters, ~330 training images per fold).
- **Melanoma in situ is the hardest class in every experiment** (AUC 0.52–0.67).

### Segmentation quality (`panderm/segmentation_stats.py`, 547 images)

| Method | Failure rule | Failures (images) | Patients with ≥ 1 failure | Failure rate differs by class? |
|---|---|:-:|:-:|:-:|
| Otsu v1 | full-frame fallback | 0.2% | 0.6% | no (p = 0.31) |
| Otsu v2 | fallback to field of view | 0.9% | 3.1% | no (p = 0.86) |
| SAM2 v1 | implausible mask (< 3% or > 80% of frame) | 38.0% | 57.1% | no (p = 0.23) |
| SAM2 v1 + preprocessing | implausible mask | 34.7% | 53.4% | no (p = 0.64) |
| SAM2 v2 | SAM mask rejected → Otsu v2 | 11.2% | 22.7% | no (p = 0.74) |

Otsu v2 and SAM2 v2 masks agree closely (median IoU 0.90, 3.5% of images below IoU 0.5) because SAM2 v2 is prompted from the Otsu v2 estimate; where they disagree, each is better on different images. No ground-truth masks yet — validation on ISIC 2018 Task 1 / PH2 is planned.

### Earlier results (Otsu v1 crops, previous machine — not re-run since the restructure)

| Model | Eval set | Balanced Acc | Macro AUC | Macro F1 |
|-------|:--------:|:------------:|:---------:|:--------:|
| PanDerm Large (frozen) — Otsu v1 crops | 163 pt | 0.569 ± 0.061 | 0.729 ± 0.033 | 0.561 ± 0.058 |
| **BioClinicalBERT — reports** | 174 pt | **0.852 ± 0.062** | **0.953 ± 0.023** | **0.852 ± 0.060** |
| Late Fusion (frozen PanDerm + BioClinicalBERT) | 156 pt | 0.739 ± 0.060 | 0.896 ± 0.036 | 0.738 ± 0.059 |

Per class (F1 / AUC, mean ± SD):

| Class | PanDerm frozen | BioClinicalBERT | Late Fusion (frozen) |
|-------|:--------------:|:---------------:|:--------------------:|
| Dysplastic Nevus | 0.556 ± 0.090 / 0.784 ± 0.057 | 0.897 ± 0.037 / 0.974 ± 0.016 | 0.767 ± 0.069 / 0.941 ± 0.019 |
| Melanoma Stage IA | 0.687 ± 0.090 / 0.813 ± 0.078 | 0.858 ± 0.081 / 0.970 ± 0.037 | 0.813 ± 0.082 / 0.933 ± 0.030 |
| Melanoma In Situ | 0.441 ± 0.140 / 0.588 ± 0.082 | 0.800 ± 0.082 / 0.914 ± 0.035 | 0.635 ± 0.087 / 0.815 ± 0.075 |

Late fusion vs PanDerm (frozen): Macro AUC +0.168 [+0.097, +0.238], Balanced Acc +0.169 [+0.100, +0.239], both p = 0.003 (paired t-test, 5 shared folds).

Backbone benchmark (frozen encoders on the same Otsu v1 crops, same folds and head): ConvNeXt-Tiny 0.731 ± 0.026 and ConvNeXt-Small 0.729 ± 0.052 Macro AUC vs PanDerm 0.729 ± 0.033 — no significant difference (all p > 0.9). The `baselines/` scripts are not included in this repository.

---

## Pipeline

```
Dermoscopy images                                       Pathology reports
      │                                                        │
      ├─ (optional) segment_lesions.py                         ▼
      │     backend: color | sam | color_v2 | sam_v2 | sam_v1_pre      report_cleaning.py
      │     → crops (segmented_cache/<backend>/)                   │
      │                                                        ▼
      ▼                                                  BioClinicalBERT
PanDerm Large ViT-L/16 (frozen)                          (mean pooling)
      │                                                        │
      ├─ extract_features.py           CLS token, raw image or crop
      ├─ extract_features_maskguided.py  lesion-weighted patch tokens of the RAW image
      │      (--pooling mean: unweighted patch mean, control)
      │                                                        │
      ▼                                                        │
patient features (1024) = mean over the patient's images       │
      │                                                        │
      ├─ evaluate.py  --mode frozen | maskguided | meanall     ├─ clinical_pipeline.py
      │   (logistic regression, nested C)                     │
      │                                                        │
      └──────────────────────┬─────────────────────────────────┘
                             ▼
                      late_fusion.py  (per-modality PCA(8) → concat → logistic regression)

Separately: panderm_finetuning.py (partial fine-tune, own head on each fold's test patients)
            → evaluate.py --mode finetune
            segmentation_stats.py (failure rates, class dependence, artifacts vs errors, mask agreement)
```

---

## Repository Structure

```
Multimodal-AI-for-Oncology/
├── config.py                          # All paths, SEGMENTATION_BACKEND, hyperparameters — edit this first
├── run_pipeline.py                    # Stage orchestrator
├── requirements.txt                   # Install CUDA PyTorch FIRST (see Quickstart)
│
├── panderm/
│   ├── prepare_data.py                # Scan dataset → manifest + patient-level 5-fold splits
│   ├── segment_lesions.py             # Segmenters (Otsu v1/v2, SAM2 v1/v2, SAM2 v1 + preprocessing) + crops + QC
│   ├── make_panderm_csv.py            # Fold CSVs for fine-tuning (images chosen by SEGMENTATION_BACKEND)
│   ├── panderm_finetuning.py          # Partial fine-tune + test predictions with the fold's own head
│   ├── extract_features.py            # Frozen CLS features (raw image or crop)
│   ├── extract_features_maskguided.py # Mask-guided patch pooling (+ --pooling mean control)
│   ├── evaluate.py                    # Metrics, summaries, per-patient predictions, plots
│   ├── segmentation_stats.py          # Segmentation statistics Q1–Q6
│   └── archive/                       # Superseded / one-off analysis scripts
│
├── clinical/
│   ├── clinical_pipeline.py           # BioClinicalBERT embeddings from pathology reports
│   ├── report_cleaning.py             # Graded leakage control (orig → termfilt → diagdrop → fact → notext)
│   ├── ocr_reports.py                 # OCR front-end for scanned PDFs (TCGA) — optional
│   └── tcga_extract.py                # Pull text layer from TCGA report PDFs — optional
│
└── fusion/
    └── late_fusion.py                 # Late fusion: image + text → logistic regression
```

---

## Quickstart

### 1. Environment (Python 3.11)

```bash
conda create -n oncology python=3.11 -y
conda activate oncology

# 1) CUDA PyTorch FIRST — a plain `pip install torch` gives a CPU-only build on Windows.
#    cu128 is required for RTX 50-series (Blackwell) GPUs; older GPUs can use cu121/cu124.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2) Everything else (numpy<2.0, timm==0.9.16, ...)
pip install -r requirements.txt
pip install "open_clip_torch<3"          # imported by PanDerm's models package; 2.x matches timm 0.9.16

# 3) SAM2 (only for the sam / sam_v2 / sam_v1_pre segmenters).
#    Official package; skip the CUDA extension build so pip does not replace the CUDA torch.
set SAM2_BUILD_CUDA=0                    # PowerShell: $env:SAM2_BUILD_CUDA="0"
pip install --no-build-isolation "git+https://github.com/facebookresearch/sam2.git"

git clone https://github.com/SiyuanYan1/PanDerm.git   # anywhere; point config.py at it
```

Check the GPU: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.
Do **not** run PanDerm's own `classification/requirements.txt` — it pins an old torch.

### 2. Configure `config.py`

```python
DATA_ROOT        = Path(r"C:\...\PhD_Data\dermoscopy")                 # DATA_ROOT/{DN,MIA,Minsitu}/{patient_id}/*.jpg
OUTPUT_DIR       = Path(r"C:\...\PhD_Data\inhouse_pipeline\outputs")
PANDERM_REPO     = Path(r"C:\...\PanDerm")
CHECKPOINT_LARGE = Path(r"C:\...\panderm_ll_data6_checkpoint-499.pth")
CLINICAL_INPUT   = Path(r"C:\...\PhD_Data\harmonized_pathology_reports.csv")
CLINICAL_DIR     = Path(r"C:\...\PhD_Data\clinical_outputs")

SEGMENTATION_BACKEND = "none"   # "none", "color", "sam", "color_v2", "sam_v2", "sam_v1_pre"
```

Every stage picks its images and its output folders from `SEGMENTATION_BACKEND`:
`none` = raw images; any other value = that backend's crops (`segmented_cache/<backend>/`) or, for mask-guided pooling, that backend's masks.

> Keep `OUTPUT_DIR` out of cloud-synced folders (OneDrive etc.) if another device syncs the same account: sync conflicts have overwritten manifests and deleted checkpoints during training.

### 3. Run

```bash
# Data (once; deterministic folds)
python run_pipeline.py --stage prepare

# Frozen PanDerm on raw images (SEGMENTATION_BACKEND = "none")
python panderm/extract_features.py
python panderm/evaluate.py --mode frozen

# Crops: set SEGMENTATION_BACKEND = "color_v2" (or another segmenter), then
python run_pipeline.py --stage segment
python panderm/extract_features.py
python panderm/evaluate.py --mode frozen

# Mask-guided pooling: SEGMENTATION_BACKEND = "sam_v2" (any segmenter)
python panderm/extract_features_maskguided.py
python panderm/evaluate.py --mode maskguided

# Control — unweighted patch mean: SEGMENTATION_BACKEND = "none"
python panderm/extract_features_maskguided.py --pooling mean
python panderm/evaluate.py --mode meanall

# Partial fine-tuning (SEGMENTATION_BACKEND = "none")
python panderm/make_panderm_csv.py
python panderm/panderm_finetuning.py            # --eval-only re-scores existing checkpoints
python panderm/evaluate.py --mode finetune

# Segmentation statistics (any backend setting)
python panderm/segmentation_stats.py

# Text and fusion
python run_pipeline.py --stage clinical_modality
python run_pipeline.py --stage fusion --no-finetune
```

---

## Stage Details

### `prepare_data.py`
Scans `DATA_ROOT/{class}/{patient_id}/`, builds `dataset_manifest.csv` with patient-level stratified 5-fold splits (`StratifiedGroupKFold`, groups = `patient_id`, seed 42) and validates there is no leakage. The manifest stores this machine's absolute paths — regenerate it after moving data.

**Output:** `OUTPUT_DIR/dataset_manifest.csv`

### `segment_lesions.py`
Segments the raw image, crops the lesion box (+10%), sets the background to black and resizes to 224 × 224. Backends:

| Backend | Method |
|---|---|
| `color` (Otsu v1) | LAB L-channel Otsu on the whole image, closing, fill holes, largest component; full-frame fallback if mask < 1% |
| `color_v2` (Otsu v2) | Masks computed at 1024 px on a DullRazor hair-removed copy; dermoscope field of view detected (dark border-connected regions removed); Otsu threshold from field-of-view pixels only; closing + opening + fill holes; component chosen by area × centrality, rejecting components mostly in the outer ring; fallback = field of view |
| `sam` (SAM2 v1) | SAM2 (`sam2.1-hiera-small`) with one positive point at the image centre; highest-scoring of 3 masks |
| `sam_v1_pre` | `sam`'s prompt and selection on `color_v2`'s preprocessing; mask clipped to the field of view (ablation) |
| `sam_v2` | SAM2 prompted with the Otsu v2 lesion box (+5%), its deepest interior point (positive) and skin points outside the box (negative); clean-up (component containing the point, fill holes); highest-scoring **plausible** mask (contains the point, 2–95% of the field, inside the field); fallback Otsu v2 → field of view |

Crops keep the original pixels; hair removal is used only to find the mask. v2 backends write per-image QC (`segmentation_qc_<backend>.csv`: hair fraction, field-of-view fraction, fallback, mask source, mask area).

**Output:** `SEGMENTED_DIR/<backend>/`, QC CSV, `segmentation_qc_montage_<backend>.png`

### `make_panderm_csv.py`
Five fold CSVs (`image`, `label`, `split`): test = fold *i*, validation = fold *i + 1*, train = the rest. Images follow `SEGMENTATION_BACKEND`; stops if any image is missing.

**Output:** `CSV_DIR/<backend>/panderm_finetuning_fold{0-4}.csv`

### `panderm_finetuning.py` — GPU
Partial fine-tune of PanDerm-L: patch embedding, CLS token and blocks 0–17 frozen; blocks 18–23 + head trained (75.6 M parameters). AdamW, peak LR 2.5e-4 with layer decay 0.65 (block 18 = 1.9e-5), 5-epoch linear warmup then per-step cosine to 1e-6, 30 epochs, batch 32, weight decay 0.05, class-weighted cross-entropy with label smoothing 0.1, drop-path 0.1 on trainable blocks only; RandomResizedCrop(0.75–1), H/V flips, 45° rotation, hue jitter; no mixup/cutmix. The epoch with the best validation balanced accuracy is kept. After training, **the fold's own head predicts only that fold's test patients**; softmax outputs are averaged per patient. Folds with an existing checkpoint are not retrained.

**Output:** `checkpoints/finetune/<backend>/results_fold{i}/` — `checkpoint-best.pth`, `training_log_fold{i}.csv`, `test_predictions_fold{i}.csv` (~5.3 GB GPU memory, ~4 min per fold on an RTX 5080)

### `extract_features.py` — GPU
Frozen PanDerm on every image (raw or crop, per `SEGMENTATION_BACKEND`); the CLS token after the final norm; averaged per patient. Loading prints `missing=2, unexpected=92` — expected (new 3-class head; self-supervised decoder discarded).

**Output:** `features/frozen/<backend>/` — `frozen_image_features.npy`, `patient_features_fold{i}.npy`, `patient_labels.npy`, `patient_group_ids.npy`, `group_fold_mapping.csv`

### `extract_features_maskguided.py` — GPU
The **raw** image goes through frozen PanDerm; all 197 output tokens are kept. The segmenter's mask (cached as PNG) gets the same Resize(256) + CenterCrop(224), is split into PanDerm's 14 × 14 grid of 16 × 16-pixel patches, and each patch's weight = its fraction of lesion pixels. Image feature = weighted average of the 196 patch tokens (all weights 1 if no lesion falls in the crop). `--pooling mean` gives every patch weight 1 (control; requires backend `none`).

**Output:** `features/maskguided/<backend>/` (+ `masks/`, `mask_weight_stats.csv`) or `features/meanall/none/`

### `evaluate.py`
`--mode frozen | maskguided | meanall`: multinomial logistic regression (`class_weight="balanced"`, C ∈ {0.01, 0.03, 0.1, 0.3, 1.0} chosen by inner 3-fold CV on the training patients). `--mode finetune`: reads the fine-tuned heads' test predictions. All modes share one `compute_metrics()`.

**Output:** `results/<mode>/<backend>/` — `kfold_results_<mode>.csv` (per fold), `kfold_summary_<mode>.csv` (mean ± SD, 95% CI and pooled, overall and per class), `pooled_report_<mode>.txt`, `patient_predictions_<mode>.csv`, confusion matrix, ROC curves, fold bars, UMAP. The attention-map step prints "failed" in frozen modes (no fold checkpoint) — expected.

### `segmentation_stats.py`
Compares all segmenters at patient level: Q1 failure rates (Wilson 95% CI), Q2 failure vs class (permutation chi-square, Cramér's V), Q3 vignette / heavy hair / fallback vs misclassification (Fisher + class-stratified permutation, Holm), Q4 mask agreement (IoU, Dice), Q5 mask size (Wilcoxon, Bland–Altman), Q6 lowest-agreement montages. Reuses cached masks; segments missing backends once. Q3 needs `patient_predictions_*.csv` (re-run `evaluate.py`).

**Output:** `stats_segmentation/`

### `clinical_pipeline.py`
BioClinicalBERT, mean-pooled over tokens, on reports cleaned at the `termfilt` level (regex-strips diagnosis / staging / invasion terms), with k-NN LOO, permutation test, length-confound check and its own 5-fold LR.

**Output:** `CLINICAL_DIR/clinical_embeddings.npy` (174, 768), `full_reports.csv`, `text_only_kfold_results.csv`, `umap_embeddings.png`

### `late_fusion.py`
Per-modality `StandardScaler → PCA(8)` (fit on train only) → concat(16) → logistic regression on the 156 patients with both modalities.

> **Known issue:** fusion reads image features from `features/<backend>/`, but `extract_features.py` now writes to `features/frozen/<backend>/`, and fine-tuned *features* are no longer produced. Fusion needs updating before it is re-run (and should combine fine-tuned head *probabilities*, not features).

**Output:** `OUTPUT_DIR/fusion_results/`

---

## Compute

Current machine: RTX 5080 (16 GB), 64 GB RAM, Windows 11.

| Stage | Time |
|---|---|
| prepare | seconds |
| segment `color_v2` / `sam_v2` | ~0.7 / ~0.8 s per image (~7 min for 547) |
| extract_features / mask-guided (masks cached) | a few minutes |
| evaluate | < 1 min |
| partial fine-tuning | ~4 min per fold, ~20 min for 5 folds |
| segmentation_stats (masks cached) | a few minutes (permutation tests) |

---

## Requirements

| Library | Version |
|---------|---------|
| Python | 3.11 |
| PyTorch / torchvision | 2.11 / 0.26 (cu128) — any CUDA build matching your GPU |
| numpy | **< 2.0** |
| timm | **0.9.16** |
| open_clip_torch | 2.x (imported by PanDerm's `models` package) |
| sam2 | Meta's official package (SAM2 segmenters only) |
| transformers | as in `requirements.txt` |
| scikit-learn / scipy / scikit-image / opencv-python | 1.x / 1.x / 0.2x / 4.x |
| umap-learn, matplotlib, seaborn, pandas, tqdm, Pillow | latest |
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
